from settings import *
from utils import *

from src.data.preprocessing import preprocess, features_to_scale, features_input

from src.train.trainer import Trainer

import pandas as pd
import numpy as np
import joblib

import torch
from torch import nn
# from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Callable

import uuid
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler



# def iterative_forecast(seq, model, steps, seq_len):
#     predicted = []
#     current_sequence = seq[:seq_len].reshape(1,seq_len,7)
#     # current_sequence = last_known[-seq_len:]
#     for k in range(steps):
#         # next_pred = model.predict(current_sequence.reshape(1, seq_len, -1))[0]
#         x_test = torch.Tensor(current_sequence).to(DEVICE)
#         y_pred = model.predict(x_test)[-1,:]

#         predicted.append(y_pred)
#         seq[seq_len+k] = np.array([seq[k+seq_len][0], *y_pred])
        
#         current_sequence = seq[k+1:k+1+seq_len].reshape(1,seq_len,7)

#     return predicted


def iterative_forecast_on_long_lat_diff(seq, model, steps, seq_len):
    preds = []
    # diff_lat_pred = []
    current_sequence = seq[:seq_len].reshape(1,seq_len,21)
    # current_sequence = last_known[-seq_len:]
    for k in range(steps):
        # next_pred = model.predict(current_sequence.reshape(1, seq_len, -1))[0]
        x_test = torch.Tensor(current_sequence).to(DEVICE)
        y_pred = model.predict(x_test)[-1,:]

        preds.append(y_pred)
        # diff_long_pred.append(y_pred[-2])
        # diff_lat_pred.append( y_pred[-1] )

        seq[seq_len+k] = np.array([seq[k+seq_len][:17], *y_pred[:-2]])
        
        current_sequence = seq[k+1:k+1+seq_len].reshape(1,seq_len,21)

    return preds


def torch_model_pipeline(
        model: nn.Module,
        do_preprocess: bool = True,
        loss: Callable = nn.MSELoss(),
        opt: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 5e-4,
        seq_len: int = 32, 
        seq_type = "n_in_1_out",
        seq_len_out: int = 1,
        verbose: bool = True,
        to_torch: bool = True,
        parallelize_seq: bool = False,
        scaler: MinMaxScaler = MinMaxScaler(),
        epochs_tr: int = 200,
        epochs_ft: int = 500
    ):
    # OPEN NEEDED `*.csv` files

    ais_train = pd.read_csv(AIS_TRAIN, sep='|')
    ais_train['time'] = pd.to_datetime(ais_train['time'])

    ais_test = pd.read_csv(AIS_TEST, sep=",")
    ais_test['time'] = pd.to_datetime(ais_test['time']) 

    if do_preprocess:

        X_train, X_val, y_train, y_val, test_set, scaler, df_lat_long, dropped_vessel_ids = preprocess(
            ais_train, 
            ais_test,
            seq_type=seq_type,
            seq_len=seq_len,
            seq_len_out=seq_len_out,
            verbose=verbose,
            to_torch=to_torch,
            parallelize_seq=parallelize_seq,
            scaler=scaler
        )

        print(f"Preprocessing ok... Number of vessels dropped: {len(dropped_vessel_ids)}")
        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(y_train)

        X_val = torch.Tensor(X_val)
        y_val = torch.Tensor(y_val)

        torch.save(X_train, LAST_PREPROCESS_FOLDER.joinpath("X_train.pt"))
        torch.save(y_train, LAST_PREPROCESS_FOLDER.joinpath("y_train.pt"))
        torch.save(X_val, LAST_PREPROCESS_FOLDER.joinpath("X_val.pt"))
        torch.save(y_val, LAST_PREPROCESS_FOLDER.joinpath("y_val.pt"))

        joblib.dump(scaler, LAST_PREPROCESS_FOLDER.joinpath("scaler")) 
        test_set.to_csv(LAST_PREPROCESS_FOLDER.joinpath("test_set.csv"))

    else:
        try:
            
            X_train = torch.load(LAST_PREPROCESS_FOLDER.joinpath("X_train.pt"), weights_only=True)
            y_train = torch.load(LAST_PREPROCESS_FOLDER.joinpath("y_train.pt"), weights_only=True)
            X_val = torch.load(LAST_PREPROCESS_FOLDER.joinpath("X_val.pt"), weights_only=True)
            y_val = torch.load(LAST_PREPROCESS_FOLDER.joinpath("y_val.pt"), weights_only=True)

            scaler = joblib.load(LAST_PREPROCESS_FOLDER.joinpath("scaler")) 
            test_set = pd.read_csv(LAST_PREPROCESS_FOLDER.joinpath("test_set.csv"))

        except:
            print(f"ERROR: File missing in {str(LAST_PREPROCESS_FOLDER)}. Now run preprocessing...")
            return torch_model_pipeline(
                model=model,
                do_preprocess=True,
                loss=loss,
                opt=opt,
                lr=lr,
                seq_len=seq_len, 
                seq_type=seq_type,
                seq_len_out=seq_len_out,
                verbose=verbose,
                to_torch=to_torch,
                parallelize_seq=parallelize_seq,
                scaler=scaler
            )


    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if torch.cuda.device_count() > 1:
    #         model = DDP(model)

    model.to(DEVICE)
    optimizer = opt(model.parameters(), lr=lr)
    trainer = Trainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        device=DEVICE,
    )
    X_train = torch.Tensor(X_train).to(DEVICE)
    y_train = torch.Tensor(y_train).to(DEVICE)

    X_val = torch.Tensor(X_val).to(DEVICE)
    y_val = torch.Tensor(y_val).to(DEVICE)

    y_train = y_train.reshape(-1, 6)
    y_val = y_val.reshape(-1, 6)

    print("Start training...")
    trainer.fit(
        X=X_train,
        y=y_train,
        # X_val=X_val,
        # y_val=y_val,
        epochs=epochs_tr,
        eval_on_test=True,
        k_folds=0,
    )

    score = trainer.eval(X_val, y_val)
    import numpy as np

    try:
        print("Score on validation set (rmse):", np.sqrt(score))
    except:
        try:
            print("Score on validation set (rmse):", np.sqrt(score.cpu().numpy()))
        except:
            print("Score ???")


    print("Start fine tuning...")
    model = trainer.best_model
    optimizer = opt(model.parameters(), lr=lr/10)
    X = torch.cat([X_train, X_val], dim=0)
    y = torch.cat([y_train, y_val], dim=0)
    final_trainer = Trainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        device=DEVICE
    )
    final_trainer.fit(
        X=X,
        y=y,
        # X_val=X_val,
        # y_val=y_val,
        epochs=epochs_ft,
        eval_on_test=True,
        k_folds=0,
        split_ratio=.95
    )
    # PREDICTION STEP

    grouped_test = test_set.groupby("vesselId")

    predictions = []

    sequence_length = seq_len

    for vessel_id, group in tqdm(grouped_test, colour="green"):
        forecast_steps = len(group['time'].values) - seq_len

        last_known_features = group[features_input].values

        # MAYBE NAN VALUES HERE (SHOULD TAKE LAST NOT NAN VALUES) - MAYBE TODO
        last_vessel_lat = df_lat_long.loc[df_lat_long['vesselId'] == vessel_id].sort_values(by='time')['latitude'].values[-1]
        last_vessel_long = df_lat_long.loc[df_lat_long['vesselId'] == vessel_id].sort_values(by='time')['longitude'].values[-1]

        preds = iterative_forecast_on_long_lat_diff(
            last_known_features, 
            last_vessel_long, 
            last_vessel_lat, 
            trainer, 
            forecast_steps, 
            sequence_length
        )
        
        group.loc[group.index[seq_len:],['cog', 'sog', 'rot', 'heading', 'long_diff', 'lat_diff']] = preds
        
        group[features_input] = scaler.inverse_transform(group[features_input])

        group.loc[group.index[seq_len:],'longitude'] = group.loc[group.index[seq_len:], 'long_diff'].values.cumsum() + group.loc[group.index[seq_len -1],'longitude']
        group.loc[group.index[seq_len:],'latitude'] = group.loc[group.index[seq_len:], 'lat_diff'].values.cumsum() + group.loc[group.index[seq_len -1],'latitude']
        # group.loc[group.index[seq_len:],'latitude'] = lat_pred
        
        predictions.append(group.copy())

    df_preds = pd.concat(predictions, ignore_index=True)

    # SUBMIT RESULT
    
    res = df_preds[["ID","longitude","latitude"]].sort_values("ID")[:51739]
    res = res.reset_index().drop(columns="index")
    res["longitude_predicted"] = res["longitude"]
    res["latitude_predicted"] = res["latitude"]
    
    res = res.drop(columns=["longitude", "latitude"])

    def make_file_name() -> str:
        file_name = str(uuid.uuid4()) + ".csv"
        print(f"Submission file name is: {file_name}")
        return file_name

    def submit(forecast: pd.DataFrame, file_name: str = None) -> None:
        sample_submission = pd.read_csv(AIS_SAMPLE_SUBMISSION)
        file_name = file_name if file_name else make_file_name()

        repertory = SUBMISSION_FODLER.joinpath(file_name)
        sample_submission = sample_submission[['ID']].merge(forecast[["ID","longitude_predicted","latitude_predicted"]], on='ID', how='left')
        try:
            sample_submission.to_csv(repertory, index=False)
        except:
            print("Error register file")
            submit(forecast)

    print("res describe")
    print(res.describe())
    submit(res)