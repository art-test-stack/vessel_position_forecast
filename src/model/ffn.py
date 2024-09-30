from settings import DEVICE
import torch
from torch import nn

from typing import Union, Callable


class FFNModel(nn.Module):
    def __init__(
            self, 
            num_features: int = 7,
            seq_len: int = 1,
            dropout: float = .1,
            layer_norm_eps: float = 0.00001,
            bias: bool = True,
            device: torch.device | str = DEVICE
        ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features * seq_len, 64, bias=bias, dropout=dropout),
            # nn.LayerNorm(64, eps=layer_norm_eps),
            nn.Sigmoid(),
            nn.Linear(64, 64, bias=bias, dropout=dropout),
            nn.Sigmoid(),
            nn.Linear(64, 32, bias=bias, dropout=dropout),
            nn.Sigmoid(),
            nn.Linear(32, 16, bias=bias, dropout=dropout),
            nn.Sigmoid(),
            nn.Linear(16, 6, bias=bias, dropout=dropout),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        len_b = x.shape[0]
        if len(x.shape) == 3:
            x = x.reshape(len_b, -1)
        
        out = self.model(x)
        return out


