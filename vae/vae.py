import torch
from torch import nn
from torch.nn import functional as F


class MIDI(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, sequence_length, last_cell_only = True):
        super(MIDI, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
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
        # the LSTM implements sequence many cells, so `x` contains the output (hidden state) of each cell
        # if we only need the last cells output we could do
        # x = x[:, -1, :]

        if self.last_cell_only:
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
        z = torch.cat([z for _ in range(self.sequence_length-1)], 1)
        zx = torch.cat((x[:, :-1, :],z), 2)
        out = self.decode(zx)
        return out, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def bce_kld_loss(recon_x, x, mu, logvar):
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