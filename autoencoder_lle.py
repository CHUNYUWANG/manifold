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


model = autoencoder([3, 50, 30, 2]).cuda()
criterion = mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

batch_size = 500
data_path = './som_swiss_roll.h5'
dataset = SwissRollDataset(data_path)
dataloader = SwissRollDataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 2000
for epoch in range(num_epochs):
    iteration = 0
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
        torch.cuda.synchronize()

        loss_recon = criterion(output, y)
        loss_topol = criterion(torch.mm(c, basis_hidden), data_hidden)
        loss = loss_recon + 10*loss_topol

        torch.cuda.synchronize()
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration += 1

    print('Epoch[{}/{}], loss:{:.4f}, loss:{:.4f}'.format(
        epoch + 1, num_epochs, loss_recon.item(), loss_topol.item()))
    dataloader.reset()

torch.save(model.state_dict(), './autoencoder_lle.pth')

dataset = SwissRollDataset(data_path)
dataloader = SwissRollDataLoader(
    dataset, batch_size=500, shuffle=False, add_basis=False)
with torch.no_grad():
    hiddens = []
    colors = []
    src_data = []
    for data in dataloader:
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
plt.show()
