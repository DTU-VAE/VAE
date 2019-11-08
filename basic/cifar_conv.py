import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from functools import reduce

batch_size = 128
epochs = 100
seed = 1
log_interval = 10
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

torch.manual_seed(seed)

transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
dataset_test = datasets.CIFAR10('./data', train=False, transform=transform_test)

classes = [1]
def stratified_sampler(labels):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [np.array(labels) == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)

train_loader = torch.utils.data.DataLoader(
    dataset_train,
    sampler=stratified_sampler(dataset_train.targets),
    batch_size=batch_size,
    #shuffle=True,
    **kwargs)
test_loader = torch.utils.data.DataLoader(
    dataset_test,
    sampler=stratified_sampler(dataset_test.targets),
    batch_size=batch_size,
    #shuffle=True,
    **kwargs)


class VAE(nn.Module):
    def __init__(self, in_shape, embedding_size):
        super(VAE, self).__init__()

        self.depth = in_shape[3]
        self.embedding_size = embedding_size

        # encode convolution layer
        self.conv1 = nn.Conv2d(self.depth, 32, 3, 1)
        self.btnm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.btnm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(0.05)
        self.conv5 = nn.Conv2d(128, 256, 3, 1)
        
        self.height, self.width = self.calc_out_shape(in_shape[1],in_shape[2], self.conv1, self.conv2, self.pool2, self.conv3, self.conv4, self.pool4, self.conv5)

        # encode linear
        self.fc1 = nn.Linear(256*self.width*self.height, 2000)
        self.fc2 = nn.Linear(2000, 1500)
        self.fc3 = nn.Linear(1500, 1000)
        self.fc4 = nn.Linear(1000, 500)

        # laten space (mean, std)
        self.fc51 = nn.Linear(500, self.embedding_size)
        self.fc52 = nn.Linear(500, self.embedding_size)
        
        # decode linear
        self.fc6 = nn.Linear(self.embedding_size, 500)
        self.fc7 = nn.Linear(500, 1000)
        self.fc8 = nn.Linear(1000, 1500)
        self.fc9 = nn.Linear(1500, 2000)
        self.fc10 = nn.Linear(2000, 256*self.width*self.height)

        # deconde convolution layer
        self.dconv1 = nn.ConvTranspose2d(256, 128, 3, 1)
        #self.dpool2 = nn.MaxUnpool2d(2, 2)
        self.dconv2 = nn.ConvTranspose2d(128, 128, 3, 1)
        self.dbtnm2 = nn.BatchNorm2d(128)
        self.dconv3 = nn.ConvTranspose2d(128, 64, 3, 1)
        #self.dpool4 = nn.MaxUnpool2d(2, 2)
        self.dconv4 = nn.ConvTranspose2d(64, 32, 3, 1)
        self.dbtnm4 = nn.BatchNorm2d(32)
        self.dconv5 = nn.ConvTranspose2d(32, self.depth, 3, 1)

        # activation function used
        self.activation = nn.ELU()

    def calc_out_shape(self, height, width, *fns):
        """
            Returns the final width, height of the matrix after the convolution layer.
        """
        ret_height = height
        ret_width  = width
        for fn in fns:
            try:
                ret_height = ((ret_height + 2*fn.padding[0] - fn.dilation[0]*(fn.kernel_size[0]-1) - 1) / fn.stride[0]) + 1
                ret_width  = ((ret_width  + 2*fn.padding[1] - fn.dilation[1]*(fn.kernel_size[1]-1) - 1) / fn.stride[1]) + 1
            except:
                ret_height = ((ret_height + 2*fn.padding - fn.dilation*(fn.kernel_size-1) - 1) / fn.stride) + 1
                ret_width  = ((ret_width  + 2*fn.padding - fn.dilation*(fn.kernel_size-1) - 1) / fn.stride) + 1

        return int(ret_height), int(ret_width)

    def encode(self, x):
        x = self.activation(self.btnm1(self.conv1(x)))
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.activation(self.btnm3(self.conv3(x)))
        x = self.activation(self.conv4(x))
        x = self.pool4(x)
        x = self.drop4(x)
        x = self.activation(self.conv5(x))

        x = x.view(-1,256*self.height*self.width)
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        
        return self.fc51(x), self.fc52(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.activation(self.fc6(z))
        z = self.activation(self.fc7(z))
        z = self.activation(self.fc8(z))
        z = self.activation(self.fc9(z))
        z = self.activation(self.fc10(z))

        z = z.view(-1, 256, self.width, self.height)

        z = self.activation(self.dconv1(z))
        z = F.interpolate(z, scale_factor = 2)
        z = self.activation(self.dbtnm2(self.dconv2(z)))
        z = self.activation(self.dconv3(z))
        z = F.interpolate(z, scale_factor = 2)
        z = self.activation(self.dbtnm4(self.dconv4(z)))
        z = self.dconv5(z)

        #return z
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

model = VAE(train_loader.dataset.data.shape, 20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    sh = train_loader.dataset.data.shape
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, sh[1]*sh[2]*sh[3]), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                sh = train_loader.dataset.data.shape
                #comparison = torch.cat([data[:n], recon_batch.view(batch_size, sh[3], sh[1], sh[2])[:n]])
                comparison = torch.cat([data[:n], recon_batch[:n]])
                save_image(comparison.cpu(), 'results_conv_cifar/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(128, 20).to(device)
            sample = model.decode(sample).cpu()
            sh = train_loader.dataset.data.shape
            #save_image(sample.view(64, sh[3], sh[1], sh[2]), 'results_conv_cifar/sample_' + str(epoch) + '.png')
            save_image(sample, 'results_conv_cifar/sample_' + str(epoch) + '.png')
