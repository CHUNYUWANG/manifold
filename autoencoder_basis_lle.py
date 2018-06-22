import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D
from swiss_roll_lle import SwissRollDataset
from swiss_roll_lle import SwissRollDataLoader
import collections


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()


class autoencoder(nn.Module):

    def __init__(self, dims):
        super(autoencoder, self).__init__()
        nhiddenLayers = len(dims) - 1
        encoder_layers = []
        for i in range(nhiddenLayers):
            encoder_layers.append(('encode_linear_{}'.format(i),
                                   nn.Linear(dims[i], dims[i + 1])))
            if i < nhiddenLayers - 1:
                encoder_layers.append(('encoder_relu_{}'.format(i),
                                       nn.ReLU(True)))
        self.encoder = nn.Sequential(collections.OrderedDict(encoder_layers))

        decoder_layers = []
        for i in range(nhiddenLayers):
            decoder_layers.append(('decode_linear_{}'.format(i),
                                   nn.Linear(dims[nhiddenLayers - i],
                                             dims[nhiddenLayers - i - 1])))
            if i < nhiddenLayers - 1:
                decoder_layers.append(('decoder_relu_{}'.format(i),
                                       nn.ReLU(True)))
        self.decoder = nn.Sequential(collections.OrderedDict(decoder_layers))

    def forward(self, x):
        s = self.encoder(x)
        y = self.decoder(s)
        return s, y


model = autoencoder([3, 500, 200, 100, 50, 30, 2]).cuda()
criterion = mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

batch_size = 500
data_path = './som_swiss_roll.h5'
dataset = SwissRollDataset(data_path)
dataloader = SwissRollDataLoader(dataset, batch_size=batch_size, shuffle=True)

testdataset = SwissRollDataset(data_path)
testdataloader = SwissRollDataLoader(
    testdataset, batch_size=500, shuffle=False, add_basis=False)

num_epochs = 40000
for epoch in range(num_epochs):
    iteration = 0
    epoch_recon_loss = 0
    epoch_topol_loss = 0
    for data in dataloader:
        x = torch.tensor(data['data']).float()
        y = torch.tensor(data['label']).float()
        c = torch.tensor(data['coeff']).float()
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        c = Variable(c).cuda()
        # ===================forward=====================
        hidden, output = model(x)
        data_hidden = Variable(
            hidden[:batch_size, :], requires_grad=True).cuda()
        basis_hidden = Variable(
            hidden[batch_size:, :], requires_grad=True).cuda()

        # ===================loss========================
        torch.cuda.synchronize()
        loss_recon = criterion(output, y)
        loss_topol = criterion(torch.mm(c, basis_hidden), data_hidden)
        loss = loss_recon + loss_topol
        torch.cuda.synchronize()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_recon_loss += loss_recon.item()
        epoch_topol_loss += loss_topol.item()
        iteration += 1

    print('Epoch[{}/{}], loss:{:.4f}, loss:{:.4f}'.format(
        epoch + 1, num_epochs, epoch_recon_loss / iteration,
        epoch_topol_loss / iteration))
    dataloader.reset()

    if np.mod(epoch, 100) == 0:
        with torch.no_grad():
            hiddens = []
            colors = []
            src_data = []
            for data in testdataloader:
                x = torch.tensor(data['data']).float()
                x = Variable(x).cuda()
                hidden, output = model(x)
                hidden = hidden.cpu().numpy()
                hiddens.append(hidden)
                color = data['color']
                colors.append(color)
                #src_data.append(data['data'])
                src_data.append(output.cpu().numpy())

        hiddens = np.concatenate(hiddens, axis=0)
        src_data = np.concatenate(src_data, axis=0)
        colors = np.concatenate(colors, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(211, projection='3d')
        ax.scatter(
            src_data[:, 0],
            src_data[:, 1],
            src_data[:, 2],
            c=colors,
            cmap=plt.cm.Spectral)

        ax.set_title("Original data")
        ax = fig.add_subplot(212)
        ax.scatter(hiddens[:, 0], hiddens[:, 1], c=colors, cmap=plt.cm.Spectral)
        plt.axis('tight')
        plt.xticks([]), plt.yticks([])
        plt.title('Projected data')
        #plt.show()
        plt.savefig('{}.jpg'.format(epoch))
        plt.close()
        testdataloader.reset()

torch.save(model.state_dict(), './autoencoder_lle.pth')
