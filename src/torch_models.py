import torch
from torch import nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=100, depth=2, activation=nn.ReLU):
        super(MLP, self).__init__()
        
        
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        self.main = nn.Sequential(
            OrderedDict([
                (f"hid_{i}", self.ResidualBlock(hidden_dim, activation))
                for i in range(depth)
            ])
        ) 
        self.tail = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.fc0(x)
        x = self.activation()(x)
        x = self.main(x)
        x = self.tail(x)
        return x
    
    class ResidualBlock(nn.Module):
        def __init__(self, dim, activation):
            super(MLP.ResidualBlock, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(dim, dim),
                activation(),
                nn.BatchNorm1d(dim),
                nn.Linear(dim, dim),
                activation(),
                nn.BatchNorm1d(dim),
            )
        
        def forward(self, x):
            output = self.main(x)
            return output + x
        
        
class MLPLogExtended(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=100, depth=2, activation=nn.ReLU):
        super(MLPLogExtended, self).__init__()
        self.main = MLP(input_dim * 2, hidden_dim, depth, activation)
    
    def forward(self, x):
        x = torch.cat([x, torch.log(x)], dim=1)
        x = self.main(x)
        return x