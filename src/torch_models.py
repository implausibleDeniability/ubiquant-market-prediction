import torch
from torch import nn
from collections import OrderedDict
from math import log10


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
        dropout=0.0,
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
                    (f"hid_{i}", ResidualBlock(hidden_dim, activation, dropout))
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


class MLPLogExtended(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=100, depth=2, activation=nn.ReLU):
        super(MLPLogExtended, self).__init__()
        self.main = MLP(input_dim * 2, hidden_dim, depth, activation)

    def forward(self, x):
        x = torch.cat([x, torch.log(x)], dim=1)
        x = self.main(x)
        return x

    
def closest_power_of_2(n):
    res = 0;
    for i in range(n, 0, -1):
        if ((i & (i - 1)) == 0):
            res = i
            break
    return res    


class MLPAE(nn.Module):
    def __init__(self, input_dim=300, bottleneck_dim=64, hidden_dim=100, mlp_depth=2, activation=nn.ReLU):
        super(MLPAE, self).__init__()
        self.input_dim = input_dim
        assert (bottleneck_dim & (bottleneck_dim-1) == 0) and bottleneck_dim != 0, "Bottleneck dimension should be power of 2"     
        self.bottleneck_dim = bottleneck_dim
        self.mlp_depth = mlp_depth
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.ae_depth = int((log10(input_dim) - log10(bottleneck_dim)) / log10(2)) - 1
        
        self.ae_encoder = self.__build_encoder()
        self.ae_decoder = self.__build_decoder()
        self.mlp = self.__build_mlp()
    
    def __build_encoder(self):
        hidden_dim = closest_power_of_2(self.input_dim)
        encoder_layers = [nn.Linear(self.input_dim, hidden_dim)]
        for i in range(self.ae_depth):
            input_dim = hidden_dim
            output_dim = int(hidden_dim / 2)
            layer = BlockAE(input_dim, output_dim, self.activation)
            encoder_layers.append(layer)
            hidden_dim = output_dim
        encoder_layers.append(nn.Linear(hidden_dim, int(hidden_dim / 2)))
        return torch.nn.Sequential(*encoder_layers)
    
    def __build_decoder(self):
        hidden_dim = self.bottleneck_dim
        decoder_layers = []
        for i in range(self.ae_depth):
            input_dim = hidden_dim
            output_dim = int(hidden_dim * 2)
            layer = BlockAE(input_dim, output_dim, self.activation)
            decoder_layers.append(layer)
            hidden_dim = output_dim
        decoder_layers.append(nn.Linear(hidden_dim, int(hidden_dim * 2)))
        decoder_layers.append(nn.Linear(int(hidden_dim * 2), self.input_dim))
        return torch.nn.Sequential(*decoder_layers)
        
    def __build_mlp(self):
        mlp_layers = [nn.Linear(self.bottleneck_dim + self.input_dim, self.hidden_dim)]
        for i in range(self.mlp_depth):
            mlp_layers.append(ResidualBlock(self.hidden_dim, self.activation, 0.0))
        mlp_layers.append(nn.Linear(self.hidden_dim, 1))
        return torch.nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        representation = self.ae_encoder(x)
        extended = torch.cat((x, representation), 1)
        predicted = self.mlp(extended)        
        mapping = self.ae_decoder(representation)
        return mapping, predicted

        
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