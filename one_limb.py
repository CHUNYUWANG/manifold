""" 
We create a limb dataset with joint angles smaller than a threshold.
This is used to validate that auto-encoders learn the boundaries.
"""

import numpy as np
import torch.utils.data as tdata


def create_limb_dataset(N, minangle, maxangle, cut_off):
    angles = np.linspace(minangle, maxangle, N)
    data = []
    label = []
    for angle in angles:
        x, y = np.cos(angle), np.sin(angle)
        data.append(np.reshape(np.array([1, 0, 0, 0, x, y]), (6,)))

        if angle > cut_off:
            xl, yl = np.cos(cut_off), np.sin(cut_off)
        else:
            xl, yl = np.cos(angle), np.sin(angle)
        label.append(np.reshape(np.array([1, 0, 0, 0, xl, yl]), (6,)))
    return np.array(data), np.array(label)


class LimbDataset(tdata.Dataset):

    def __init__(self, N, minangle, maxangle, cut_off):
        self.data, self.label = create_limb_dataset(N, minangle, maxangle,
                                                    cut_off)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx, :]
        label = self.label[idx, :]
        sample = {'data': data, 'label': label}
        return sample
