import torch
from torch import nn


class BlockAE(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super(BlockAE, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            activation()
        )
        
    def forward(self, x):
        out = self.main(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, dim, activation, dropout):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(dim, dim),
            activation(),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            activation(),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        output = self.main(x)
        return output + x