import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import pretty_midi
import vae

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
#kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

epochs = 1
batch_size = 10
sequence_length = 50

midi_dataset = vae.midi_dataloader.MIDIDataset('..\data', sequence_length=sequence_length, fs=16, year=2004, add_limit_tokens=False, binarize=False, save_pickle=True)
dataloader = DataLoader(midi_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


class MIDI(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, last_cell_only = False):
        super(MIDI, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.last_cell_only = last_cell_only

        # encode rnn
        self.rnn1 = nn.LSTM(self.input_size,self.hidden_size,num_layers=1,batch_first=True,dropout=0,bidirectional=False)

        # encode linear
        self.fc1 = nn.Linear(self.hidden_size, 500)

        # laten space (mean, std)
        self.fc21 = nn.Linear(500, self.embedding_size)
        self.fc22 = nn.Linear(500, self.embedding_size)
        
        # decode linear
        self.fc3 = nn.Linear(self.embedding_size, 500)
        self.fc4 = nn.Linear(500, self.hidden_size)

        # deconde rnn
        self.drnn1 = nn.LSTM(self.hidden_size,self.input_size,num_layers=1,batch_first=True,dropout=0,bidirectional=False)

        # activation function used
        self.activation = nn.ReLU()

    def encode(self, x):
        x, (h, c) = self.rnn1(x)

        # `x` has shape (batch, sequence, hidden_size)
        # the LSTM implement sequence many cells, so `x` contains the output (hidden state) of each cell
        # if we only need the last cells output we could do
        # x = x[:, -1, :]

        if self.last_cell_only:
            # UNDONE: implement many-to-one case
            raise NotImplementedError('Need to implement the case where only the last cell\'s output is used.')
            #x = x[:,-1,:].view(-1, self.hidden_size)
        else:
            x = x.contiguous().view(-1, self.hidden_size)

        x = self.activation(self.fc1(x))
        
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.activation(self.fc3(z))
        z = self.activation(self.fc4(z))

        z = z.view(batch_size, sequence_length, self.hidden_size)

        z, (h, c) = self.drnn1(z)

        return z
        #return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

model = MIDI(88,300,40).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #TODO: Decide which loss function to use

    #BCE = F.cross_entropy(recon_x, x, reduction='sum')
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    #BCE2 = F.mse_loss(recon_x, x, reduction='mean')
    #BCE = F.ctc_loss(F.log_softmax(recon_x), x.int(), recon_x.shape, x.shape, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader),
                loss.item() / len(data)))
            if batch_idx == 50: return

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader.dataset)))


def test(epoch):
    model.eval()
    #TODO: Implement test function

#    test_loss = 0
#    with torch.no_grad():
#        for i, (data, _) in enumerate(test_loader):
#            data = data.to(device)
#            recon_batch, mu, logvar = model(data)
#            test_loss += loss_function(recon_batch, data, mu, logvar).item()
#            if i == 0:
#                n = min(data.size(0), 8)
#                sh = train_loader.dataset.data.shape
#                #comparison = torch.cat([data[:n], recon_batch.view(batch_size, sh[3], sh[1], sh[2])[:n]])
#                comparison = torch.cat([data[:n], recon_batch[:n]])
#                save_image(comparison.cpu(), 'results_conv_cifar/reconstruction_' + str(epoch) + '.png', nrow=n)

#    test_loss /= len(test_loader.dataset)
#    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(500, model.embedding_size).to(device)
            sample = model.decode(sample).cpu()
            sample = sample.contiguous().view(88,-1)
            sample = torch.sigmoid(sample)
            sample = sample * 100
            sample = sample.type(torch.ByteTensor)
            sample = torch.where(sample > 60, sample, torch.ByteTensor([0]))

            # convert piano roll to midi
            program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
            midi_from_proll = vae.midi_utils.piano_roll_to_pretty_midi(sample, fs = 16, program = program)
            midi_from_proll.write(f'sample_{epoch}.midi')