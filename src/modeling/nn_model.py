import torch
import torch.nn as nn
import torch.nn.functional as F


class RiidModel(nn.Module):
    def __init__(self, num_columns):
        super(RiidModel, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_columns)
        self.dense1 = nn.Linear(num_columns, 2048)

        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.dense2 = nn.Linear(2048, 1048)

        self.batch_norm3 = nn.BatchNorm1d(1048)
        self.dense3 = nn.Linear(1048, 1)

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = F.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = F.sigmoid(self.dense3(x)).squeeze()

        return x
