from settings import *
from utils import *

from src.data.preprocessing import preprocess, features_input

from src.train.trainer import Trainer

import pandas as pd
import numpy as np
import joblib

import torch
from torch import nn

from typing import Dict, Union

from datetime import datetime
import uuid


def iterative_forecast(seq, model, steps, sequence_length):
    predicted = []
    current_sequence = seq[:sequence_length].reshape(1,sequence_length,7)
    # current_sequence = last_known[-sequence_length:]
    for k in range(steps):
        # next_pred = model.predict(current_sequence.reshape(1, sequence_length, -1))[0]
        X_test = torch.Tensor(current_sequence).to(DEVICE)
        y_pred = model(X_test)[0,0,:]

        next_pred = y_pred.view(6).cpu().numpy()
        predicted.append(next_pred)

        seq[k+1,1:] = next_pred
        # Update current_sequence by appending next prediction
        current_sequence = seq[k+1,].view(1,sequence_length,7)
    
    return predicted


def main(seq_len, do_preprocess):
    # OPEN NEEDED `*.csv` files
    if do_preprocess:
        ais_train = pd.read_csv(AIS_TRAIN, sep='|')
        ais_train['time'] = pd.to_datetime(ais_train['time'])

        ais_test = pd.read_csv(AIS_TEST, sep=",")
        ais_test['time'] = pd.to_datetime(ais_test['time']) 

        X_train, X_val, y_train, y_val, test_set, scaler = preprocess(
            ais_train, 
            ais_test,
            seq_type="n_in_1_out",
            seq_len=seq_len,
            seq_len_out=1,
            verbose=True,
            to_torch=True
        )

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
            assert False, "Do preprocessing first"

    dim_ffn = 2048
    d_model = 512
    transformer_decoder_params = {
        "d_model": d_model,
        "nhead": 8,
        # "num_encoder_layers": 6,
        # "num_decoder_layers": 2,
        "dim_feedforward": dim_ffn,
        "dropout": 0.1,
        # "activation": str | ((Tensor) -> Tensor) = F.relu,
        "layer_norm_eps": 0.00001,
        "batch_first": True,
        "norm_first": False,
        # "bias": True,
        "device": DEVICE,
    }

    class DecoderModel(nn.Module):
        def __init__(
                self,
                decoder_params: Dict[int,Union[int, float, bool]] = transformer_decoder_params, 
                num_features: int = 7, 
                num_outputs: int = 6, 
                num_layers: int = 1,
                act_out: nn.Module | None = None
            ) -> None:
            super().__init__()
            self.emb_layer = nn.Linear(num_features, d_model)
            dec_layer = nn.TransformerDecoderLayer(**decoder_params)
            self.model = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
            self.ffn = nn.Linear(dim_ffn, num_outputs)
            self.act_out = act_out # nn.Sigmoid()
            
        def forward(self, x):
            emb = self.emb_layer(x)
            out = self.model(emb, emb)
            if self.act_out:
                return self.act_out(self.ffn(out))
            return self.ffn(out)


    model = DecoderModel()
    trainer = Trainer(
        model=model,
        loss=nn.MSELoss(),
        optimizer=torch.optim.Adam(params=model.parameters()),
        device=DEVICE
    )


    X_train = torch.Tensor(X_train).to(DEVICE)
    y_train = torch.Tensor(y_train).to(DEVICE)

    X_val = torch.Tensor(X_val).to(DEVICE)
    y_val = torch.Tensor(y_val).to(DEVICE)

    trainer.fit(
        X=X_train,
        y=y_train,
        # X_val=X_val,
        # y_val=y_val,
        epochs=500,
        eval_on_test=True
    )

    y_pred = model(X_val)

    score = trainer.metric(y_val, y_pred)

    try:
        print("Score on validation set (rmse):", np.sqrt(score))
    except:
        print("Score on validation set:", np.sqrt(score))

    # PREDICTION STEP

    grouped_test = test_set.groupby("vesselId")

    predictions = []

    sequence_length = seq_len

    for vessel_id, group in grouped_test:
        forecast_steps = len(group['time'].values) - 1

        last_known_features = group[features_input].values

        # last_known_features = scaler.transform(group[scaled_features].values)
        future_preds = iterative_forecast(last_known_features, trainer, forecast_steps, sequence_length)
        
        # Store the predictions

        df_pred = pd.DataFrame(scaler.inverse_transform(future_preds), columns=features_input)
        df_pred['time'] = group['time'].iloc[1:].values
        df_pred["vesselId"] = vessel_id
        predictions.append(df_pred)

    df_preds = pd.concat(predictions, ignore_index=True)


    # SUBMIT RESULT

    res = pd.merge(ais_test, df_preds[["vesselId","time", "latitude", "longitude"]],on=["time", "vesselId"], how="left")
    res["longitude_predicted"] = res["longitude"]
    res["latitude_predicted"] = res["latitude"]
    # res["id"] = res["ID"]
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

    submit(res)

if __name__ == "__main__":
    seq_len = 48
    do_preprocess = True
    main(seq_len, do_preprocess)