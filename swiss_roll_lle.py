""" 
We create a limb dataset with joint angles smaller than a threshold.
This is used to validate that auto-encoders learn the boundaries.
"""

import numpy as np
import torch
import torch.utils.data as tdata
from sklearn import datasets
import h5py


class SwissRollDataset(tdata.Dataset):

    def __init__(self, datapath):
        """
        add_basis: concatenate data with basis in each batch
        """
        with h5py.File(datapath, 'r') as f:
            self.data = np.array(f['data'])
            self.label = np.array(self.data)
            self.basis = np.array(f['basis'])
            self.coeff = np.array(f['coeff'])
            self.color = np.array(f['color'])

    def __len__(self):
        return self.data.shape[0]

    def getitem(self, idx):
        data = self.data[idx, :]
        label = self.label[idx, :]
        color = self.color[idx]
        coeff = self.coeff[idx, :]
        sample = {'data': data, 'label': label, 'color': color, 'coeff': coeff}
        return sample


class SwissRollDataLoader(object):

    def __init__(self, dataset, batch_size=1, shuffle=False, add_basis=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.add_basis = add_basis

        self.ndata = len(dataset)
        self.nbatches = self.ndata / batch_size
        self.cur_batch = 0

        if shuffle:
            self.idx = np.random.permutation(self.ndata)
        else:
            self.idx = np.arange(self.ndata)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_batch < self.nbatches:
            idx = self.idx[self.cur_batch * self.batch_size:
                           (self.cur_batch + 1) * self.batch_size]
            sample = self.dataset.getitem(idx)

            if self.add_basis:
                sample['data'] = np.concatenate(
                    [sample['data'], self.dataset.basis], axis=0)
                sample['label'] = np.concatenate(
                    [sample['label'], self.dataset.basis], axis=0)
            self.cur_batch += 1
            return sample
        else:
            raise StopIteration

    def reset(self):
        self.idx = np.random.permutation(self.ndata)
        self.cur_batch = 0


if __name__ == '__main__':
    dataset = SwissRollDataset('./som_swiss_roll.h5')
    dataloader = SwissRollDataLoader(dataset)
    cur = 0
    for epoch in range(10):
        for data in dataloader:
            print(cur)
            cur = cur + 1
        dataloader.reset()
