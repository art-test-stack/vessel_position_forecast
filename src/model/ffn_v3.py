from settings import DEVICE
import torch
from torch import nn

from typing import Union, Callable


class FFNModel(nn.Module):
    def __init__(
            self, 
            dim_in: int = 7,
            seq_len: int = 1,
            dropout: float = .1,
            layer_norm_eps: float = 0.00001,
            bias: bool = True,
            device: torch.device | str = DEVICE,
            **kwargs
        ) -> None:
        super().__init__()

        self.main_long = nn.Sequential(
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
            nn.Linear(16, 1, bias=bias),
            nn.Dropout(dropout),
            # nn.LayerNorm(1, eps=layer_norm_eps),
            # nn.Sigmoid(),
        )
        self.main_lat = nn.Sequential(
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
            nn.Linear(16, 1, bias=bias),
            nn.Dropout(dropout),
            # nn.LayerNorm(1, eps=layer_norm_eps),
            # nn.Sigmoid(),
        )
        
        for layer in self.main_long:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        for layer in self.main_lat:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        len_b = x.shape[0]
        if len(x.shape) == 3:
            x = x.reshape(len_b, -1)
        
        long = self.main_long(x)
        lat = self.main_lat(x)
        return torch.cat([long, lat], dim=1)


# main1 = nn.Sequential(
#     nn.Linear(num_features * seq_len, 128, bias=bias),
#     nn.Dropout(dropout),
#     # nn.LayerNorm(64, eps=layer_norm_eps),
#     nn.LayerNorm(128),
#     nn.ReLU(),
#     nn.Linear(128, 64, bias=bias),
#     nn.Dropout(dropout),
#     nn.LayerNorm(64),
#     # nn.ReLU(),
#     # nn.Linear(64, 64, bias=bias),
#     # nn.Dropout(dropout),
#     # nn.LayerNorm(64),
#     nn.ReLU(),
#     nn.Linear(64, 32, bias=bias),
#     nn.Dropout(dropout),
#     nn.LayerNorm(32),
#     nn.ReLU(),
#     nn.Linear(32, 16, bias=bias),
#     nn.Dropout(dropout),
#     nn.LayerNorm(16),
#     nn.ReLU(),
#     nn.Linear(16, dim_out, bias=bias),
#     nn.Dropout(dropout),
#     nn.LayerNorm(dim_out),
#     # nn.Sigmoid(),
# )