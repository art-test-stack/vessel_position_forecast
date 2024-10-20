from settings import DEVICE

import torch
from torch import nn

from typing import Dict, Union, Callable


# transformer_params = {
#     "d_model": d_model,
#     "nhead": 8,
#     # "num_encoder_layers": 6,
#     "num_decoder_layers": 2,
#     "dim_feedforward": dim_ffn,
#     "dropout": 0.1,
#     # "activation": str | ((Tensor) -> Tensor) = F.relu,
#     "custom_encoder": None,
#     "custom_decoder": None,
#     "layer_norm_eps": 0.00001,
#     "batch_first": False,
#     "norm_first": False,
#     "bias": True,
#     "device": None,
# }

activation_dec: Union[str | Callable[[torch.Tensor], torch.Tensor]] = nn.SiLU()


dim_ffn = 128
d_model = 64

transformer_decoder_params = {
    "d_model": d_model,
    "nhead": 8,
    # "num_encoder_layers": 6,
    # "num_decoder_layers": 2,
    "dim_feedforward": dim_ffn,
    "dropout": 0.1,
    "activation": activation_dec,
    "layer_norm_eps": 0.00001,
    "batch_first": True,
    "norm_first": False,
    # "bias": True,
    "device": DEVICE,
}

params = {
    "num_layers": 1,
    "dim_ffn": 128,
    "d_model": 64,
    "nhead": 8,
    "dropout": .1,
    "layer_norm_eps": 0.00001,
    "tf_norm_first": False,
    "bias": True,
    "act_dec": nn.SiLU(),
    "act_out": None,
}


class EncoderModel(nn.Module):
    def __init__(
            self, 
            num_features: int = 7, 
            dim_out: int = 6, 
            seq_len: int = 32,
            num_layers: int = 1,
            dim_ffn: int = 128,
            d_model: int = 32,
            nhead: int = 8,
            dropout: float = .5,
            layer_norm_eps: float = 0.00001,
            tf_norm_first: bool = False,
            bias: bool = True,
            act_dec: Union[str | Callable[[torch.Tensor], torch.Tensor]] = nn.ReLU(),
            act_out: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None = None,
            compute_mean: bool = False,
            device: torch.device | str = DEVICE
        ) -> None:
        super().__init__()
        self.compute_mean = compute_mean

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ffn,
            dropout=dropout,
            activation=act_dec,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=tf_norm_first,
            bias=bias,
        )
        # self.model = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        # self.ffn = nn.Linear(d_model, num_outputs, bias=bias)
        self.te = nn.Sequential(
            nn.Linear(num_features, d_model, bias=bias),
            nn.TransformerEncoder(dec_layer, num_layers=num_layers),
        )
        self.main = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=bias),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2, bias=bias),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model // 2, dim_out, bias=bias),
        )
        self.act_out = act_out # nn.Sigmoid()
        
    def forward(self, x):
        len_b, len_s, _ = x.shape
        x = self.te(x)[:, -1, :].view(len_b, -1) if not self.compute_mean else torch.mean(out, dim=1).reshape(len_b, -1)
        out = self.main(x)
        
        if self.act_out is not None:
            return self.act_out(out)
        return out


    # def get_pad_mask(self, seq: torch.Tensor):

    #     pad_idx = 0 # self.padding_idx
    #     pad_mask = (seq != pad_idx).unsqueeze(-2)

    #     _, len_s = seq.size()
    #     subsequent_mask = (1 - torch.triu(
    #         torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    #     return pad_mask & subsequent_mask