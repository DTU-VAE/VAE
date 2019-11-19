import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import pretty_midi
import vae


parser = argparse.ArgumentParser(description='VAE MIDI')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--sequence-length', type=int, default=50, metavar='N',
                    help='sequence length of input data to LSTM (default: 50)')
parser.add_argument('--colab', action='store_true', default=False,
                    help='indicates whether script is running on Google Colab')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--bootstrap', type=str, default='',
                    help='specifies the path to the model.tar to load the model from')
args = parser.parse_args()


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
#kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

if not Path(args.bootstrap).is_file():
    args.bootstrap = ''

if args.colab:
    midi_dataset = vae.midi_dataloader.MIDIDataset('data/maestro-v2.0.0', sequence_length=args.sequence_length, fs=16, year=2004, add_limit_tokens=False, binarize=True, save_pickle=False)
else:
    midi_dataset = vae.midi_dataloader.MIDIDataset('../data', sequence_length=args.sequence_length, fs=16, year=2004, add_limit_tokens=False, binarize=True, save_pickle=True)

train_sampler, test_sampler, validation_sampler = vae.midi_dataloader.split_dataset(midi_dataset, test_split=0.15, validation_split=0.15)
train_loader      = DataLoader(midi_dataset, batch_size=args.batch_size, sampler=train_sampler,      drop_last=True)
test_loader       = DataLoader(midi_dataset, batch_size=args.batch_size, sampler=test_sampler,       drop_last=True)
validation_loader = DataLoader(midi_dataset, batch_size=args.batch_size, sampler=validation_sampler, drop_last=True)


class MIDI(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, last_cell_only = True):
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
        #self.fc3 = nn.Linear(self.input_size+self.embedding_size, 500)
        #self.fc4 = nn.Linear(500, self.input_size)

        # deconde rnn
        self.drnn1 = nn.LSTM(self.input_size+self.embedding_size,self.input_size,num_layers=1,batch_first=True,dropout=0,bidirectional=False)

        # activation function used
        self.activation = nn.ReLU()

    def encode(self, x):
        x, (h, c) = self.rnn1(x)

        # `x` has shape (batch, sequence, hidden_size)
        # the LSTM implement sequence many cells, so `x` contains the output (hidden state) of each cell
        # if we only need the last cells output we could do
        # x = x[:, -1, :]

        if self.last_cell_only:
            # TEST: implement many-to-one case
            x = x[:,-1,:]
        else:
            x = x.contiguous().view(-1, self.hidden_size)

        x = self.activation(self.fc1(x))
        
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def decode(self, zx):
        #z = self.activation(self.fc3(z))
        #z = self.activation(self.fc4(z))

        #z = z.view(batch_size, sequence_length, self.hidden_size)

        if self.training:
            x, (h, c) = self.drnn1(zx)
        else:
            #TODO: implement sample generation

            # ONLY TEMPORARLY HERE
            x, (h, c) = self.drnn1(zx)

        return x

    def forward(self, x):
        mu, logvar = self.encode(x[:, 1:, :])
        z = self.reparameterize(mu, logvar)
        z = z.view(10,1,64)
        z = torch.cat([z for _ in range(args.sequence_length-1)], 1)
        zx = torch.cat((x[:, :-1, :],z), 2)
        out = self.decode(zx)
        return out, mu, logvar

model = MIDI(88,300,64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #TODO: Decide which loss function to use
    #TODO: BCE reduction mean/sum?

    #BCE = F.cross_entropy(recon_x, x, reduction='sum')
    #BCE = F.mse_loss(recon_x, x, reduction='sum')
    #BCE2 = F.mse_loss(recon_x, x, reduction='mean')
    #BCE = F.ctc_loss(F.log_softmax(recon_x), x.int(), recon_x.shape, x.shape, reduction='sum')

    # using mean of both BCE and KLD
    #BCE = F.binary_cross_entropy(torch.sigmoid(recon_x), x[:, 1:, :], reduction='sum')
    #BCE /= args.batch_size
    #TODO: should I call torch.bernoulli() on the sigmoid recon_x?
    BCE = F.binary_cross_entropy(torch.sigmoid(recon_x), x[:, 1:, :], reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #TODO: check if KLD needs to be normalised or not
    #KLD /= args.batch_size

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), args.batch_size*len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader)))

    if args.colab:
        save_path = f'model_states/model_epoch_{epoch}.tar'
    else:
        save_path = f'../model_states/model_epoch_{epoch}.tar'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
            }, save_path)


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
    c_epoch = 0

    # load the model parameters from the saved file (.tar extension)
    if args.bootstrap:
        checkpoint = torch.load(args.bootstrap)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        c_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('Bootstrapping model from {}\Continuing training from epoch: {}\n'.format(args.bootstrap, c_epoch+1))

    for epoch in range(c_epoch+1, (c_epoch + args.epochs + 1)):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample_z = torch.randn(1, 1, model.embedding_size)
            sample_x = torch.zeros(1, 1, model.input_size) #TODO: check if this is good or not
            sample   = torch.cat([sample_x, sample_z], 2).to(device)

            sample = model.decode(sample).cpu()
            sample = torch.bernoulli(torch.sigmoid(sample)) #TODO: do I call torch.bernoulli() on this sample?
            sample = sample * 100
            sample = sample.contiguous().view(88,-1) #TODO: use contiguous()? or reshape?

            # convert piano roll to midi
            program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
            midi_from_proll = vae.midi_utils.piano_roll_to_pretty_midi(sample, fs = 16, program = program)
            midi_from_proll.write(f'sample_{epoch}.midi')