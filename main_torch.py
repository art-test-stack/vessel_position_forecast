from settings import *
from utils import *

from src.model.ffn import FFNModel
from src.train.pipeline_torch import torch_model_pipeline

import torch
from torch import nn
from datetime import datetime
import uuid
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler



if __name__ == "__main__":
    seq_len = 3
    do_preprocess = False

    dim_in = 20
    dim_out = 7
    preprocess_file = Path("data/preprocessed_with_sincos_heading_seq_len_3/")
    if not preprocess_file.exists():
        preprocess_file.mkdir()
        do_preprocess = True
    # TODO: ADD DROPOUT ARG

    torch_model_pipeline(
        model = FFNModel,
        do_preprocess = do_preprocess,
        loss = nn.MSELoss(),
        opt = torch.optim.AdamW,
        lr = 5e-6,
        seq_len = seq_len, 
        seq_type = "n_in_1_out",
        seq_len_out = 1,
        verbose = True,
        to_torch = True,
        parallelize_seq = True,
        scaler = StandardScaler(),
        epochs_tr=500,
        epochs_ft=500,
        skip_training=False,
        dropout=.4,
        preprocess_folder = preprocess_file
    )