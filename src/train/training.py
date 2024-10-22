import torch
from torch import nn
from typing import Dict, Callable
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from settings import DEVICE, MODEL_FOLDER
from src.train.trainer import Trainer
from uuid import uuid4

def torch_train_part(
        model: nn.Module,
        model_params: Dict | None,
        training_params: Dict | None,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        X_val: np.ndarray | torch.Tensor,
        y_val: np.ndarray | torch.Tensor,
        skip_training: bool = False
    ):  
    dim_in = X_train.shape[-1]
    dim_out = y_train.shape[-1]
    # model params = {num_features=dim_in, dim_out=dim_out, seq_len=seq_len, dropout=dropout}
    model_params["dim_out"] = dim_out
    model_params["dim_in"] = dim_in
    model = model(**model_params)

    model.to(DEVICE)

    training_params["opt"] = training_params["opt"](model.parameters(), lr=training_params["lr"])
    trainer = Trainer(
        model=model,
        device=DEVICE,
        **training_params
    )
    
    X_train = torch.Tensor(X_train).to(DEVICE) if isinstance(X_train, np.ndarray) else X_train.to(DEVICE)
    y_train = torch.Tensor(y_train).to(DEVICE) if isinstance(y_train, np.ndarray) else y_train.to(DEVICE)

    X_val = torch.Tensor(X_val).to(DEVICE) if isinstance(X_val, np.ndarray) else X_val.to(DEVICE)
    y_val = torch.Tensor(y_val).to(DEVICE) if isinstance(y_val, np.ndarray) else y_val.to(DEVICE)

    if not skip_training:
        print("Training Main Model...")
        trainer.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=training_params["epochs"],
            eval_on_test=training_params["eval_on_test"],
            k_folds=0,
        )

    score = trainer.eval(X_val, y_val)

    try:
        print("Score on validation set (rmse):", np.sqrt(score) / y_val.shape[0])
    except:
        try:
            print("Score on validation set (rmse):", np.sqrt(score.cpu().numpy())/ y_val.shape[0])
        except:
            print("Score ???")

    return trainer


def xgb_train_part(
        model: xgb.XGBRegressor,
        model_params: Dict | None,
        training_params: Dict | None,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        X_val: np.ndarray | torch.Tensor,
        y_val: np.ndarray | torch.Tensor,
        skip_training: bool = False
    ):
    
    dim_in = X_train.shape[-1]
    dim_out = y_train.shape[-1]
    seq_len = X_train.shape[1]

    X_train = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
    X_val = X_val.numpy() if isinstance(X_val, torch.Tensor) else X_val
    y_train = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
    y_val = y_val.numpy() if isinstance(y_val, torch.Tensor) else y_val

    xgb_reg = model(
        device="cuda"
    )
    grid_search = GridSearchCV(
        xgb_reg,
        param_grid=model_params,
        **training_params
    )

    X_train = X_train.reshape(-1, dim_in * seq_len)
    X_val = X_val.reshape(-1, dim_in * seq_len)
    y_train = y_train.reshape(-1, dim_out)
    y_val = y_val.reshape(-1, dim_out)

    if not skip_training:
        grid_search.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=4
        )

    try:
        eval_metric = "rmse"

        results = grid_search.best_estimator_.evals_result()
        train_errors = results['validation_0'][eval_metric]
        test_errors = results['validation_1'][eval_metric]

        import matplotlib.pyplot as plt
        file_name = MODEL_FOLDER.joinpath(f"xgb_{str(uuid4())}.png")
        fig = plt.figure()
        plt.plot(train_errors, label='Train')
        plt.plot(test_errors, label='Test')
        plt.xlabel('Boosting Rounds')
        plt.ylabel(eval_metric)
        plt.legend()
        plt.title('Learning Curves')
        plt.savefig(file_name)
        plt.close(fig)

        print(f"Learning curves plot on {file_name}")
    except:
        print("Error plotting learning curves")

    try:
        model = grid_search.best_estimator_
        print("Best model params:", grid_search.best_params_)
        score = model.score(X_val, y_val)
        print("Score on validation set for best model:", score)
        model = grid_search.best_estimator_
        print("Best model params:", grid_search.best_params_)
        score = model.score(X_val, y_val)
        print("Score on validation set for best model:", score)
    except:
        print("Score ???")

    # last_best_params = {
    #     'early_stopping_rounds': 50, 
    #     'eta': 0.05, 
    #     'gamma': 10, 
    #     'learning_rate': 0.01, 
    #     'max_depth': 10, 
    #     'min_child_weight': 15, 
    #     'n_estimators': 2000, 
    #     'subsample': 1.0
    # }

       # best_params = {
    #   'early_stopping_rounds': 50, 
    #   'eta': 0.05, 
    #   'gamma': 5, 
    #   'max_depth': 5, 
    #   'min_child_weight': 10, 
    #   'n_estimators': 2000, 
    #   'subsample': 1.0
    # }

    # best_params_2 = {
    #     'colsample_bytree': 0.7, 
    #     'early_stopping_rounds': 50, 
    #     'eta': 0.05, 
    #     'gamma': 5, 
    #     'max_depth': 5, 
    #     'min_child_weight': 7, 
    #     'n_estimators': 3000, 
    #     'subsample': 1.0
    # }

    # model = xgb.XGBRegressor(
    #     device="cuda",
    #     **best_params
    #     # **model_params,
    #     # eval_metric=mean_absolute_error,
    #     )
    
    if not skip_training:
        print("Start training...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )

    
    try:
        score = model.score(X_val, y_val)
        print("Score on validation set (rmse):", np.sqrt(score))
    except:
        try:
            score = model.score(X_val, y_val)
            print("Score on validation set (rmse):", np.sqrt(score.cpu().numpy()))
        except:
            print("Score ???")