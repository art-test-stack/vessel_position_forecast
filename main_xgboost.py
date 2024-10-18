from settings import *
from utils import *
import xgboost as xgb
from src.model.ffn import FFNModel
from src.train.pipeline_xgb import xgb_model_pipeline

import torch
from torch import nn
from datetime import datetime
import uuid
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler



if __name__ == "__main__":
    seq_len = 1
    do_preprocess = False

    dim_in = 20
    dim_out = 7
    preprocess_file = Path(f"data/preprocessed_with_sincos_heading_seq_len_{seq_len}/")
    if not preprocess_file.exists():
        preprocess_file.mkdir()
        do_preprocess = True
    # TODO: ADD DROPOUT ARG

    params = {
    # 'n_estimators': 5000,
        'gamma': 0.5,
        'subsample': 0.6,
        'n_estimators': 5000,
        'min_child_weight':  15,
        'colsample_bytree': 0.8,
        'max_depth': 4,
        'eta': 0.005,
        'refresh_leaf': 1,
        # "early_stopping_rounds": 50,
    }
    
    xgb_model_pipeline(
        model_params = params,
        do_preprocess = do_preprocess,
        loss = nn.MSELoss(),
        opt = torch.optim.AdamW,
        # lr = 5e-6,
        seq_len = seq_len, 
        seq_type = "basic",
        seq_len_out = 1,
        verbose = True,
        to_torch = True,
        parallelize_seq = True,
        scaler = StandardScaler(),
        # epochs_tr=500,
        # epochs_ft=500,
        # skip_training=True,
        dropout=.4,
        preprocess_folder = preprocess_file
    )