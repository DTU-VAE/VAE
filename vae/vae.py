import torch
from torch import nn
from torch.nn import functional as F


#class MIDI(nn.Module):
#    def __init__(self, input_size, hidden_size, embedding_size, sequence_length, last_cell_only = True):
#        super(MIDI, self).__init__()

#        self.input_size = input_size
#        self.hidden_size = 88
#        self.embedding_size = embedding_size
#        self.sequence_length = sequence_length
#        self.last_cell_only = last_cell_only

#        # encode rnn
#        self.rnn1 = nn.LSTM(self.input_size,self.hidden_size,num_layers=1,batch_first=True,dropout=0,bidirectional=False)
#        self.fc4 = nn.Linear(self.input_size, self.input_size)
#        self.activation = nn.ReLU()

#    def forward(self, x):
#        x, (h, c) = self.rnn1(x)
#        x = self.activation(self.fc4(x))
#        x1 = torch.sigmoid(x)
#        x2 = torch.softmax(x)
#        return x, 0, 0






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
        
        # deconde rnn
        self.drnn1 = nn.LSTM(self.input_size+self.embedding_size,self.hidden_size,num_layers=1,batch_first=True,dropout=0,bidirectional=False)

        # decode linear
        self.fc4 = nn.Linear(self.hidden_size, self.input_size)

        # activation function used
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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
        x, (h, c) = self.drnn1(zx)

        #TEST: add linear layer to hidden states
        return self.sigmoid(self.fc4(x))

    def forward(self, x):
        mu, logvar = self.encode(x[:, 1:, :])
        z = self.reparameterize(mu, logvar)
        z = torch.unsqueeze(z,1) # add a third dimension (middle) to be able to concatenate with x
        z = torch.cat([z for _ in range(self.sequence_length-1)], 1)
        zx = torch.cat((x[:, :-1, :], z), 2)
        out = self.decode(zx)
        return out, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def bce_kld_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x[:, 1:, :], reduction='none')
    BCE = torch.sum(BCE, (1,2)) # sum over 2nd and 3rd dimensions (keeping it separate for each batch)
    BCE = torch.mean(BCE) # average over batch losses
    #BCE = F.binary_cross_entropy(recon_x, x[:, 1:, :], reduction='sum') # taken out in favour of above
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1) # sum over 2nd dimension
    KLD = torch.mean(KLD) # average over batch losses
    
    return BCE + KLD

def simple_loss(recon_x, x, _, __):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='none')
    BCE = torch.sum(BCE, (1,2)) # sum over 2nd and 3rd dimensions (keeping it separate for each batch)
    BCE = torch.mean(BCE) # average over batch losses
    return BCE