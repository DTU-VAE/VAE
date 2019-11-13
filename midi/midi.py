# from vae import midi_dataloader

import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
#kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

batch_size = 2
chunk_size = 10


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

        z = z.view(batch_size, chunk_size, self.hidden_size)

        z, (h, c) = self.drnn1(z)

        #return z
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

model = MIDI(128,300,20).to(device)

input = torch.randn(batch_size, chunk_size, 128)
ret = model.forward(input)
print(ret[0].shape)

#optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
#def loss_function(recon_x, x, mu, logvar):
#    sh = train_loader.dataset.data.shape
#    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, sh[1]*sh[2]*sh[3]), reduction='sum')
#    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

#    # see Appendix B from VAE paper:
#    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#    # https://arxiv.org/abs/1312.6114
#    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#    return BCE + KLD


#def train(epoch):
#    model.train()
#    train_loss = 0
#    for batch_idx, (data, _) in enumerate(train_loader):
#        data = data.to(device)
#        optimizer.zero_grad()
#        recon_batch, mu, logvar = model(data)
#        loss = loss_function(recon_batch, data, mu, logvar)
#        loss.backward()
#        train_loss += loss.item()
#        optimizer.step()
#        if batch_idx % log_interval == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(train_loader.dataset),
#                100. * batch_idx / len(train_loader),
#                loss.item() / len(data)))

#    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


#def test(epoch):
#    model.eval()
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


#if __name__ == "__main__":
#    for epoch in range(1, epochs + 1):
#        train(epoch)
#        test(epoch)
#        with torch.no_grad():
#            sample = torch.randn(128, 20).to(device)
#            sample = model.decode(sample).cpu()
#            sh = train_loader.dataset.data.shape
#            #save_image(sample.view(64, sh[3], sh[1], sh[2]), 'results_conv_cifar/sample_' + str(epoch) + '.png')
#            save_image(sample, 'results_conv_cifar/sample_' + str(epoch) + '.png')
