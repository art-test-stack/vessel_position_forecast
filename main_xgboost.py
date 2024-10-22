from settings import *
from utils import *
import xgboost as xgb
from src.model.ffn import FFNModel
from src.train.pipeline_v1 import pipeline

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
    preprocess_file = Path(f"data/preprocessed_new_rot_seq_len_{seq_len}/")
    if not preprocess_file.exists():
        preprocess_file.mkdir()
        do_preprocess = True

    model_params = {
        'n_estimators': [2000, 3000, 5000],
        'gamma': [5, 10],
        # 'gamma': [0.5, 1, 5, 10],
        'subsample': [1.0],
        # 'subsample': [0.6, 1.0],
        'max_depth': [4, 5, 10, 15, 35, 50],
        'eta': [ 0.05 ],
        # 'eta': [ 0.005, 0.01, 0.05],
        # 'n_estimators': [ 3000, 4000 ],
        'min_child_weight': [5, 7, 10, 15],
        # 'min_child_weight': [3, 5, 7, 10],
        # 'colsample_bytree': [.7, 0.6, .5],
        'early_stopping_rounds': [50],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    training_params = {
        "cv": 5,
        "n_jobs": -1,
        "verbose": 4,
        "scoring": "neg_mean_squared_error",
    }
    pipeline(
        model=xgb.XGBRegressor,
        model_params=model_params,
        training_params=training_params,
        do_preprocess=do_preprocess,
        seq_len=seq_len, 
        seq_type="n_in_1_out",
        parallelize_seq=True,
        scaler=StandardScaler(),
        skip_training=False,
        preprocess_folder=preprocess_file,
        verbose=True,
    )