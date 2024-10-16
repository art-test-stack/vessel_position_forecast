from settings import *
from utils import *

from src.model.ffn import FFNModel
from src.train.pipeline import torch_model_pipeline

import torch
from torch import nn
from datetime import datetime
import uuid
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler



if __name__ == "__main__":
    seq_len = 32
    do_preprocess = False

    model = FFNModel(seq_len=seq_len, num_features=19)

    torch_model_pipeline(
        model = model,
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
        epochs_tr=1,
        epochs_ft=1,
        skip_training=False
    )