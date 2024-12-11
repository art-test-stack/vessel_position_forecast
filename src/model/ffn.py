from settings import DEVICE
import torch
from torch import nn

from typing import Union, Callable


class FFNModel(nn.Module):
    def __init__(
            self, 
            dim_in: int = 7,
            dim_out: int = 6,
            seq_len: int = 1,
            dropout: float = .1,
            layer_norm_eps: float = 0.00001,
            bias: bool = True,
            device: torch.device | str = DEVICE,
            **kwargs
        ) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(dim_in * seq_len, 64, bias=bias),
            nn.Dropout(dropout),
            nn.Sigmoid(),
            nn.Linear(64, 64, bias=bias),
            nn.Dropout(dropout),
            nn.Sigmoid(),
            nn.Linear(64, 32, bias=bias),
            nn.Dropout(dropout),
            nn.Sigmoid(),
            nn.Linear(32, 16, bias=bias),
            nn.Dropout(dropout),
            nn.Sigmoid(),
            nn.Linear(16, dim_out, bias=bias),
            nn.Dropout(dropout),
            nn.LayerNorm(dim_out, eps=layer_norm_eps),
            # nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor):
        len_b = x.shape[0]
        if len(x.shape) == 3:
            x = x.reshape(len_b, -1)
        
        out = self.main(x)
        return out
