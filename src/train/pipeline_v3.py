from settings import *
from utils import *

from src.data.preprocessing import preprocess, features_to_scale, features_input

from src.train.training import torch_train_part, xgb_train_part
from src.train.trainer import Trainer

from src.test.missing_features import BaseMissingFeaturesHandler, MissingFeaturesHandler

import pandas as pd
import numpy as np
import joblib

from sklearn.multioutput import MultiOutputRegressor
import torch
from torch import nn
# from torch.nn.parallel import DistributedDataParallel as DDP
import xgboost as xgb

from typing import Dict

import uuid
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler



def iterative_forecast_v3(
        seq, 
        model,
        missing_features_handler, 
        steps, 
        seq_len,
        dim_in,
    ):
    dim_missing = missing_features_handler.dim_missing
    preds = []
    current_sequence = seq[:seq_len].reshape(1, seq_len, dim_in)

    for k in range(steps):
        last_seq_item = missing_features_handler(current_sequence)

        seq[seq_len+k] = np.concatenate((seq[seq_len+k][:dim_in-dim_missing], last_seq_item), axis=None)
        current_sequence = seq[k+1:seq_len+k+1,:].reshape(1, seq_len, dim_in)

        preds = model.predict(current_sequence)[-1,:]

        preds.append(preds)

        # seq[seq_len+k] = np.concatenate((seq[k+seq_len][:dim_in - dim_out + 2], y_pred[:-2]), axis=None)
        
        # current_sequence = seq[k+1:k+1+seq_len].reshape(1, seq_len, dim_in)
    return preds


def pipeline(
        model: nn.Module | xgb.XGBModel,
        model_params: Dict | None = None,
        training_params: Dict | None = None,
        do_preprocess: bool = True,
        seq_len: int = 32, 
        seq_type = "n_in_1_out",
        seq_len_out: int = 1,
        parallelize_seq: bool = False,
        scaler: MinMaxScaler = MinMaxScaler(),
        skip_training: bool = False,
        preprocess_folder: Path | str = LAST_PREPROCESS_FOLDER,
        verbose: bool = True,
    ):

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
            to_torch=isinstance(model(), nn.Module),
            parallelize_seq=parallelize_seq,
            scaler=scaler,
            preprocess_folder=preprocess_folder
        )
    else:
        try:
            
            X_train = torch.load(preprocess_folder.joinpath("X_train.pt"), weights_only=True)
            y_train = torch.load(preprocess_folder.joinpath("y_train.pt"), weights_only=True)
            X_val = torch.load(preprocess_folder.joinpath("X_val.pt"), weights_only=True)
            y_val = torch.load(preprocess_folder.joinpath("y_val.pt"), weights_only=True)

            scaler = joblib.load(preprocess_folder.joinpath("scaler")) 
            test_set = pd.read_csv(preprocess_folder.joinpath("test_set.csv"))
            df_lat_long = pd.read_csv(preprocess_folder.joinpath("df_lat_long.csv"))
            assert X_train.shape[1] == seq_len, "Sequence length mismatch"

        except:
            print(f"ERROR: File missing in {str(preprocess_folder)}. I will do preprocessing anyway...")
            return pipeline(
                model=model,
                do_preprocess=True,
                model_params=model_params,
                training_params=training_params,
                seq_len=seq_len, 
                seq_type = seq_type,
                seq_len_out=seq_len_out,
                parallelize_seq=parallelize_seq,
                scaler=scaler,
                skip_training=skip_training,
                preprocess_folder=preprocess_folder,
                verbose=verbose,
            )

    dim_in = X_train.shape[-1]
    dim_out = y_train.shape[-1]

    y_train = y_train.reshape(-1, dim_out)
    y_val = y_val.reshape(-1, dim_out)

    # mfh = MissingFeaturesHandler()
    trainer_mfh = Trainer(
        BaseMissingFeaturesHandler(),
        lr=5e-3,
        batch_size=1024,
        # verbose = False
    ) 
    mfh = MultiOutputRegressor(trainer_mfh, n_jobs=-1)
    if not skip_training:
        print("Training Missing Values Model...")
        X = np.concatenate((X_train, X_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)
        fit_params = {
            "X_train": X,
            "y_train": y,
            "epochs": 100,
            "eval_on_test": True,
            "k_folds": 1,
        }
        # mfh.fit(X_train, y_train[:,:-2], X_val=X_val, y_val=y_val[:,:-2]) # Last values for lat and long are not used yet (v3)
        mfh.fit(X_train, y_train[:,:-2], **fit_params) # Last values for lat and long are not used yet (v3)

    if isinstance(model(), nn.Module):
        model = torch_train_part(
            model=model,
            model_params=model_params,
            training_params=training_params,
            X_train=X_train,
            y_train=y_train[:,-2],
            X_val=X_val,
            y_val=y_val[:,-2],
            skip_training=skip_training
        )
        # model_long = torch_train_part(
        #     model=model,
        #     model_params=model_params,
        #     training_params=training_params,
        #     X_train=X_train,
        #     y_train=y_train[:,-2],
        #     X_val=X_val,
        #     y_val=y_val[:,-2],
        #     skip_training=skip_training
        # )
        # model_lat = torch_train_part(
        #     model=model,
        #     model_params=model_params,
        #     training_params=training_params,
        #     X_train=X_train,
        #     y_train=y_train[:,-1],
        #     X_val=X_val,
        #     y_val=y_val[:,-1],
        #     skip_training=skip_training
        # )
    
    elif isinstance(model(), xgb.XGBModel):
        raise ValueError("XGBoost instance not supported yet - I will implement it later")
        model = xgb_train_part(
            model=model,
            model_params=model_params,
            training_params=training_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            skip_training=skip_training
        )
    else:
        raise ValueError("Model type not supported")
    
    grouped_test = test_set.groupby("vesselId")

    predictions = []

    for vessel_id, group in tqdm(grouped_test, colour="green"):
        forecast_steps = len(group['time'].values) - seq_len

        # last_known_features = get_missing_features(group, features_input, seq_len)
        last_known_features = group[features_input].values

        # MAYBE NAN VALUES HERE (SHOULD TAKE LAST NOT NAN VALUES) - MAYBE TODO
        last_vessel_lat = df_lat_long.loc[df_lat_long['vesselId'] == vessel_id].sort_values(by='time')['latitude'].values[-1]
        last_vessel_long = df_lat_long.loc[df_lat_long['vesselId'] == vessel_id].sort_values(by='time')['longitude'].values[-1]


        preds = iterative_forecast_v3(
            seq=last_known_features,
            model=model,
            missing_features_handler=mfh,
            steps=forecast_steps, 
            seq_len=seq_len,
            dim_in=dim_in,
        )
        
        group.loc[group.index[seq_len:],['long_diff', 'lat_diff']] = preds
        
        group[features_to_scale] = scaler.inverse_transform(group[features_to_scale])

        group.loc[group.index[seq_len:],'longitude'] = group.loc[group.index[seq_len:], 'long_diff'].values.cumsum() + last_vessel_long # group.loc[group.index[seq_len -1],'longitude']
        group.loc[group.index[seq_len:],'latitude'] = group.loc[group.index[seq_len:], 'lat_diff'].values.cumsum() + last_vessel_lat # group.loc[group.index[seq_len -1],'latitude']
        
        # group.loc[group.index[seq_len:],'latitude'] = lat_pred
        
        predictions.append(group.copy())

    df_preds = pd.concat(predictions, ignore_index=True)
    df_preds['longitude'] = df_preds['longitude'].apply(lambda x: (x + 180) % 360 - 180)
    df_preds['latitude'] = df_preds['latitude'].apply(lambda x: (x + 90) % 180 - 90)
    # SUBMIT RESULT
    
    res = df_preds[["ID","longitude","latitude"]].sort_values("ID")[:51739]
    res = res.reset_index().drop(columns="index")
    res["longitude_predicted"] = res["longitude"]
    res["latitude_predicted"] = res["latitude"]
    
    res = res.drop(columns=["longitude", "latitude"])

    def make_file_name() -> str:
        file_name = str(uuid.uuid4()) + ".csv"
        # print(f"Submission file name is: {file_name}")
        return file_name

    def submit(forecast: pd.DataFrame, file_name: str = None) -> None:
        sample_submission = pd.read_csv(AIS_SAMPLE_SUBMISSION)
        file_name = file_name if file_name else make_file_name()

        repertory = SUBMISSION_FODLER.joinpath(file_name)
        sample_submission = sample_submission[['ID']].merge(forecast[["ID","longitude_predicted","latitude_predicted"]], on='ID', how='left')
        try:
            sample_submission.to_csv(repertory, index=False)
            print(f"Repertory file name is: {repertory}")
        except:
            print("Error register file")
            submit(forecast)

    if res.isna().sum() > 0:
        print("ERROR: NaN values in submission")
        return

    print("res describe")
    print(res.describe())
    submit(res)

    print("OK - Pipeline finished")