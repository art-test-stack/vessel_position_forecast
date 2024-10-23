from src.train.trainer import Trainer
from src.data.preprocessing import features_missing, features_input

import torch
from torch import nn

import numpy as np
from typing import Callable, Dict, List
from pathlib import Path

class BaseMissingFeaturesHandler(nn.Module):
    def __init__(
            self, 
            dim_in: int = len(features_input), 
            hidden_size: int = 8, 
            num_layers: int = 2,
            dim_out: int = 1,
            dropout: float = 0.2,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(
        #     dim_in,
        #     hidden_size, 
        #     num_layers, 
        #     batch_first=True,
        #     dropout=dropout
        # )
        self.layer_1 = nn.Linear(dim_in * 8, hidden_size)
        self.main = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(dropout),
            nn.Sigmoid(),
            nn.Linear(hidden_size // 2, dim_out),
            nn.Dropout(dropout),
            # nn.LayerNorm(dim_out),
            # nn.Sigmoid(),
        )
        # for name, param in self.lstm.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.xavier_normal_(param)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # out, _ = self.lstm(x, (h0, c0))
        # out = self.main(out[:, -1, :].reshape(x.size(0), -1)).reshape(-1)

        b_size = x.shape[0]
        if len(x.shape) > 2:
            x = x.reshape(b_size, -1)
        x = self.layer_1(x)
        out = self.main(x).reshape(b_size, 1)

        return out


class MissingFeaturesHandler:
    def __init__(
            self,
            # features_to_handle: Dict[str, int] = features_missing, # feature_name: index
            features_to_handle: List[str] = features_missing, # feature_name: index
            # strategy: str = "one_for_each",
            models: Dict[str, Callable] = {}
        ):
        folder = Path("missing_features/")
        if not folder.exists():
            folder.mkdir()
        self.features_to_handle = features_to_handle
        self.dim_missing = len(features_to_handle)
        self.models = models or { 
            feature: Trainer(
                BaseMissingFeaturesHandler(),
                name = folder.joinpath(f"_{feature}"),
                lr=5e-4,
                batch_size=32,
                # verbose = False
            ) 
            # if isinstance(models[feature], nn.Module)
            # else models[feature]
            for feature in features_to_handle
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        for idx, feature in enumerate(self.features_to_handle):
            print(f"Training feature handler model for feature {feature}")
            model = self.models[feature]
            model.fit(X_train, y_train[:,idx], X_val=X_val, y_val=y_val[:,idx], epochs=100)

    def predict_features(self, X) -> np.ndarray:
        preds_features = []
        for feature in self.features_to_handle:
            model = self.models[feature]
            preds_features.append(model.predict(X))
        return np.array(preds_features)

    def __call__(self,X) -> np.ndarray:
        return self.predict_features(X)
    
