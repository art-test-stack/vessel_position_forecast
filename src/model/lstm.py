from src.data.preprocessing import features_input
from settings import DEVICE

import torch
from torch import nn

import numpy as np
from typing import Callable, Dict, List
from pathlib import Path

class LSTMPredictor(nn.Module):
    def __init__(
            self, 
            dim_in: int = len(features_input), 
            hidden_size: int = 64, 
            num_layers: int = 2,
            dim_out: int = 5,
            dropout: float = 0.6,
            **kwargs
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            dim_in,
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout
        )
        
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(dropout),
            nn.Sigmoid(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.Dropout(dropout),
        )

        self.main = nn.ModuleList([ 
            nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Dropout(dropout),
            nn.Sigmoid(),
            nn.Linear(hidden_size // 4, 1),
            nn.Dropout(dropout),
            ) for _ in range(dim_out)
        ])
        self.dim_out = dim_out

        # [ self.main[k].to(DEVICE) for k in range(dim_out) ]

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))

        out = self.linear(out[:, -1, :])
        out = [ layer(out).reshape(-1) for layer in self.main ]

        return out

    
