import torch
from torch import nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=100, depth=2, activation=nn.ReLU):
        super(MLP, self).__init__()

        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        self.main = nn.Sequential(
            OrderedDict(
                [
                    (f"hid_{i}", ResidualBlock(hidden_dim, activation))
                    for i in range(depth)
                ]
            )
        )
        self.tail = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = self.activation()(x)
        x = self.main(x)
        x = self.tail(x)
        return x


class EmbedMLP(nn.Module):
    def __init__(
        self,
        input_dim=300,
        hidden_dim=100,
        depth=2,
        activation=nn.ReLU,
        num_embeddings=3775,
        embedding_dim=64,
    ):
        """Same MLP, but takes the 0th feature and applies it as an embedding
        See the first two lines of forward call for details
        """

        super(EmbedMLP, self).__init__()

        self.embedder = nn.Embedding(num_embeddings, embedding_dim)
        self.fc0 = nn.Linear(input_dim - 1 + embedding_dim, hidden_dim)
        self.activation = activation
        self.main = nn.Sequential(
            OrderedDict(
                [
                    (f"hid_{i}", ResidualBlock(hidden_dim, activation))
                    for i in range(depth)
                ]
            )
        )
        self.tail = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedding = self.embedder(x[:, 0].long())
        x = torch.cat([embedding, x[:, 1:]], dim=1)

        x = self.fc0(x)
        x = self.activation()(x)
        x = self.main(x)
        x = self.tail(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim, activation):
        super(ResidualBlock, self).__init__()
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
