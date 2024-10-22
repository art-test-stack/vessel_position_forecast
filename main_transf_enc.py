from settings import *
from utils import *

from src.model.transformer_enc import EncoderModel
from src.train.pipeline_v1 import pipeline

import torch
from torch import nn
from datetime import datetime
import uuid
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler



if __name__ == "__main__":
    seq_len = 16
    do_preprocess = False

    dim_in = 20
    dim_out = 7
    preprocess_file = Path(f"data/preprocessed_last_rot_{seq_len}/")
    # preprocess_file = LAST_PREPROCESS_FOLDER
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
        "lr": 5e-5,
        "opt": torch.optim.AdamW,
        "loss": nn.MSELoss(reduction="sum"),
        "skip_training": False,
        "eval_on_test": True,
    }

    pipeline(
        model=EncoderModel,
        model_params=model_params,
        training_params=training_params,
        do_preprocess=do_preprocess,
        seq_len=seq_len, 
        seq_type="n_in_1_out",
        seq_len_out=1,
        scaler=StandardScaler(),
        parallelize_seq=True,
        preprocess_folder=preprocess_file,
        verbose=True,
    )