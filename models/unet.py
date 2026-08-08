import torch
import torch.nn as nn

from models.embedding import PositionalEncoding

class BasicDiscreteTimeModel(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.lin_in = nn.Linear(2, d_model)
        self.lin_out = nn.Linear(d_model, 2)
        self.blocks = nn.ParameterList(
            [DiscreteTimeResidualBlock(d_model=d_model) for _ in range(n_layers)]
        )

    def forward(self, x, t):
        x = self.lin_in(x)
        for block in self.blocks:
            x = block(x, t)
        return self.lin_out(x)

class DiscreteTimeResidualBlock(nn.Module):


    def __init__(self, d_model: int, maxlen: int = 512):
        super().__init__()
        self.d_model = d_model
        self.emb = PositionalEncoding(d_model=d_model, maxlen=maxlen)
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x, t):
        return self.norm(x + self.lin2(self.act(self.lin1(x + self.emb(t)))))