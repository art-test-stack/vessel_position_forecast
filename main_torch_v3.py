from settings import *
from utils import *

from src.model.ffn_v2 import FFNModelV2
from src.train.pipeline_v3 import pipeline
from src.train.loss import MultiOutputLoss

import torch
from torch import nn

from sklearn.preprocessing import MinMaxScaler, StandardScaler



if __name__ == "__main__":
    seq_len = 8
    do_preprocess = False

    dim_in = 20
    dim_out = 7
    preprocess_file = Path(f"data/preprocessed_v2_seq_{seq_len}/")

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
        "lr": 5e-4,
        "opt": torch.optim.Adam,
        "loss": MultiOutputLoss(n_outputs=2, loss=nn.MSELoss(reduction="sum")),
        "eval_on_test": True,
        "early_stopping_rounds": 100,
        "early_stopping_min_delta": 1e-4,
        "batch_size": 2048,
    }
    
    pipeline(
        model=FFNModelV2,
        model_params=model_params,
        training_params=training_params,
        do_preprocess=do_preprocess,
        seq_len=seq_len, 
        seq_type="n_in_1_out",
        seq_len_out=1,
        scaler=StandardScaler(),
        parallelize_seq=True,
        skip_training=False,
        preprocess_folder=preprocess_file,
        verbose=True,
    )