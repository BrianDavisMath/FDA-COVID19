import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, in_size, dim, n_res_blocks):
        super().__init__()

        self.layers = [nn.Linear(in_size, dim)]
        self.layers += [ResidualBlock(dim) for i in range(n_res_blocks)]
        self.layers += [nn.Linear(dim, 1)]
        
        self.fwd = nn.Sequential(*self.layers)
        return

    def forward(self, x):
        return self.fwd(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, batch_norm=True):
        super().__init__()
        
        self.layers = []
        for i in range(2):
            self.layers += [nn.Linear(dim, dim), nn.ReLU()]
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(dim))
        
        self.fwd = nn.Sequential(*self.layers)
        return

    def forward(self, x):
        return x + self.fwd(x)