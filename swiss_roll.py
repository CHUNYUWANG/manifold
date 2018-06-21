""" 
We create a limb dataset with joint angles smaller than a threshold.
This is used to validate that auto-encoders learn the boundaries.
"""

import numpy as np
import torch.utils.data as tdata
from sklearn import datasets


class SwissRollDataset(tdata.Dataset):

    def __init__(self, N):
        self.data, self.color = datasets.samples_generator.make_swiss_roll(N)
        self.label = self.data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx, :]
        label = self.label[idx, :]
        color = self.color[idx]
        sample = {'data': data, 'label': label, 'color': color}
        return sample




