import numpy as np
import torch
from torch.utils.data import Dataset


class LpLoss(object):
    def __init__(self, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        assert p > 0

        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class TemporalDataset(Dataset):
    def __init__(self, data, snapshot_length=20):
        self.data = data
        self.N, self.T, self.C, self.H, self.W = data.shape
        self.snapshot_length = snapshot_length
        self.indices = [
            (n, t)
            for n in range(self.N)
            for t in range(self.T - snapshot_length + 1)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        n, start = self.indices[idx]
        return self.data[n, start : start + self.snapshot_length]


def get_grid(H, W):
    x = np.linspace(0, 1, H)
    y = np.linspace(0, 1, W)

    x, y = np.meshgrid(x, y)
    x = x.T
    y = y.T

    grid = torch.tensor(np.concatenate((x[None], y[None]), axis=0), dtype=torch.float32)

    return grid
