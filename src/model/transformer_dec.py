from settings import DEVICE

import torch
from torch import nn

from typing import Dict, Union, Callable


dim_ffn = 128
d_model = 64

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

class DecoderModel(nn.Module):
    def __init__(
            self,
            decoder_params: Dict[int,Union[int, float, bool]] = transformer_decoder_params, 
            num_features: int = 7, 
            num_outputs: int = 6, 
            num_layers: int = 1,
            act_out: nn.Module | None = None
        ) -> None:
        super().__init__()
        self.emb_layer = nn.Linear(num_features, d_model)
        dec_layer = nn.TransformerDecoderLayer(**decoder_params)
        self.model = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.ffn = nn.Linear(dim_ffn, num_outputs)
        self.act_out = act_out # nn.Sigmoid()
        
    def forward(self, x):
        emb = self.emb_layer(x)
        out = self.model(emb, emb)
        if self.act_out:
            return self.act_out(self.ffn(out))
        return self.ffn(out)
    

    # def get_pad_mask(self, seq: torch.Tensor):

    #     pad_idx = 0 # self.padding_idx
    #     pad_mask = (seq != pad_idx).unsqueeze(-2)

    #     _, len_s = seq.size()
    #     subsequent_mask = (1 - torch.triu(
    #         torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    #     return pad_mask & subsequent_mask