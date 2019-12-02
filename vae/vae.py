import torch
from torch import nn
from torch.nn import functional as F


class MIDI(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, sequence_length):
        super(MIDI, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length

        self.reset_cells()

        # encode rnn
        self.rnn1 = nn.LSTM(self.input_size,self.hidden_size,num_layers=2,batch_first=True,dropout=0,bidirectional=True)

        # encode linear
        #self.fc1 = nn.Linear(self.hidden_size, 500)

        # laten space (mean, std)
        linear_in_size = self.hidden_size
        if self.rnn1.bidirectional:
            linear_in_size *= 2
        self.fc21 = nn.Linear(linear_in_size, self.embedding_size)
        self.fc22 = nn.Linear(linear_in_size, self.embedding_size)
        
        # deconde rnn
        self.drnn1 = nn.LSTM(self.input_size+self.embedding_size,self.hidden_size,num_layers=2,batch_first=True,dropout=0,bidirectional=False)

        # decode linear
        self.fc4 = nn.Linear(self.hidden_size, self.input_size)

        # activation function used
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        if self.h_en is not None and self.c_en is not None:
            x, (self.h_en, self.c_en) = self.rnn1(x, (self.h_en, self.c_en))
        else:
            x, (self.h_en, self.c_en) = self.rnn1(x)

        self.h_en = self.h_en.detach()
        self.c_en = self.c_en.detach()

        # `x` has shape (batch, sequence, hidden_size)
        # the LSTM implements sequence many cells, so `x` contains the output (hidden state) of each cell
        # if we only need the last cells output we could do
        # x = x[:, -1, :]

        if self.rnn1.bidirectional:
            x = x.view(x.shape[0], x.shape[1], int(x.shape[2]/self.hidden_size), self.hidden_size)
            x_0 = x[:,-1,0,:] # forward direction
            x_1 = x[:,-1,1,:] # backward direction
            x = torch.cat([x_0,x_1], 1)
        else:
            x = x[:, -1, :]

        #x = self.activation(self.fc1(x))
        
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def decode(self, zx):
        if self.h_de is not None and self.c_de is not None:
            x, (self.h_de, self.c_de) = self.drnn1(zx, (self.h_de, self.c_de))
        else:
            x, (self.h_de, self.c_de) = self.drnn1(zx)

        self.h_de = self.h_de.detach()
        self.c_de = self.c_de.detach()

        return self.sigmoid(self.fc4(x))

    def forward(self, x):
        mu, logvar = self.encode(x[:, 1:, :])
        z = self.reparameterize(mu, logvar)
        z = torch.unsqueeze(z,1) # add a third dimension (middle) to be able to concatenate with x
        z = torch.cat([z for _ in range(self.sequence_length-1)], 1)
        mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < 0.2
        mask = torch.unsqueeze(mask,2)
        mask = torch.cat([mask for _ in range(88)], 2)
        mask = mask.float().to(x.device)
        x = (1 - mask) * x
        #x = (1 - mask) * x + mask * torch.zeros_like(x)
        zx = torch.cat((x[:, :-1, :], z), 2)
        out = self.decode(zx)
        return out, mu, logvar

    def reset_cells(self):
        self.h_en = None
        self.c_en = None
        self.h_de = None
        self.c_de = None


# Reconstruction + KL divergence losses summed over all elements and batch
def bce_kld_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x[:, 1:, :], reduction='none')
    BCE = torch.sum(BCE, (1,2)) # sum over 2nd and 3rd dimensions (keeping it separate for each batch)
    BCE = torch.mean(BCE) # average over batch losses
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1) # sum over 2nd dimension
    KLD = torch.mean(KLD) # average over batch losses
    
    return BCE + KLD


class SimpleMIDI(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, sequence_length, last_cell_only = True):
        super(MIDI, self).__init__()

        self.input_size = input_size
        self.hidden_size = 88
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.last_cell_only = last_cell_only

        # encode rnn
        self.rnn1 = nn.LSTM(self.input_size,self.hidden_size,num_layers=1,batch_first=True,dropout=0,bidirectional=False)
        self.fc4 = nn.Linear(self.input_size, self.input_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x, (h, c) = self.rnn1(x)
        x = self.activation(self.fc4(x))
        x1 = torch.sigmoid(x)
        x2 = torch.softmax(x)
        return x, 0, 0


def simple_loss(recon_x, x, _, __):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='none')
    BCE = torch.sum(BCE, (1,2)) # sum over 2nd and 3rd dimensions (keeping it separate for each batch)
    BCE = torch.mean(BCE) # average over batch losses

    return BCE