from settings import *
from utils import *

from src.model.ffn import FFNModel
from src.train.pipeline_v2 import pipeline

import torch
from torch import nn
from datetime import datetime
import uuid
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler



if __name__ == "__main__":
    seq_len = 8
    do_preprocess = False

    dim_in = 20
    dim_out = 7
    preprocess_file = Path(f"data/preprocessed_v3_seq_{seq_len}/")

    if not preprocess_file.exists():
        preprocess_file.mkdir()
        do_preprocess = True
    # TODO: ADD DROPOUT ARG

    model_params = {
        "seq_len": seq_len,
        "dropout": .4
    }
    training_params = {
        "epochs": 1000,
        "lr": 5e-3,
        "opt": torch.optim.Adam,
        "loss": nn.MSELoss(reduction="sum"),
        "eval_on_test": True,
        "early_stopping_rounds": 100,
        "early_stopping_min_delta": 1e-4,
    }
    
    pipeline(
        model=FFNModel,
        model_params=model_params,
        training_params=training_params,
        do_preprocess=do_preprocess,
        seq_len=seq_len, 
        seq_type="n_in_1_out",
        seq_len_out=1,
        scaler=MinMaxScaler(),
        parallelize_seq=True,
        skip_training=False,
        preprocess_folder=preprocess_file,
        verbose=True,
    )