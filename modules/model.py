import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, in_size, dim, n_res_blocks):
        super().__init__()

        self.layers = [ResidualBlock(dim) for i in range(n_res_blocks)] + [nn.Linear(dim, 1)]
        self.fwd = nn.Sequential(*self.layers)
        return

    def forward(self, x):
        return self.fwd(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, batch_norm=False):
        super().__init__()
        
        if batch_norm:
            raise NotImplementedError
            
        self.layers = nn.Sequential(nn.Linear(dim, dim),
                                    nn.ReLU(),
                                    nn.Linear(dim, dim),
                                    nn.ReLU(),
                                    )
        return

    def forward(self, x):
        return x + self.layers(x)