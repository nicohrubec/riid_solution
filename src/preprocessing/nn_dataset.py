import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class RiidDataset(Dataset):
    def __init__(self, df, targets, mode='train'):
        self.mode = mode
        self.data = df
        if mode == 'train':
            self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train':
            x = self.data[idx]
            t = np.array(self.targets[idx])
            return torch.from_numpy(x).float(), torch.from_numpy(t).float()
        elif self.mode == 'test':
            return torch.from_numpy(self.data[idx]).float(), 0
