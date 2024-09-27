from settings import *
from utils import *

from src.data.preprocessing import preprocess, features_input

from src.train.trainer import Trainer

import pandas as pd
import numpy as np
import joblib

import torch
from torch import nn

from typing import Dict, Union, Callable

from datetime import datetime
import uuid
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler



def iterative_forecast(seq, model, steps, sequence_length):
    predicted = []
    current_sequence = seq[:sequence_length].reshape(1,sequence_length,7)
    # current_sequence = last_known[-sequence_length:]
    for k in range(steps):
        # next_pred = model.predict(current_sequence.reshape(1, sequence_length, -1))[0]
        x_test = torch.Tensor(current_sequence).to(DEVICE)
        y_pred = model.predict(x_test)[0,0,:]

        predicted.append(y_pred)
        seq[seq_len+k] = np.array([seq[k+1][0], *y_pred])
        
        current_sequence = seq[k+1:k+1+seq_len].reshape(1,seq_len,7)
    print(np.array(predicted).shape)
    return predicted


def main(seq_len, do_preprocess):
    # OPEN NEEDED `*.csv` files

    ais_train = pd.read_csv(AIS_TRAIN, sep='|')
    ais_train['time'] = pd.to_datetime(ais_train['time'])

    ais_test = pd.read_csv(AIS_TEST, sep=",")
    ais_test['time'] = pd.to_datetime(ais_test['time']) 

    if do_preprocess:

        X_train, X_val, y_train, y_val, test_set, scaler = preprocess(
            ais_train, 
            ais_test,
            seq_type="n_in_1_out",
            seq_len=seq_len,
            seq_len_out=1,
            verbose=True,
            to_torch=True,
            parallelize_seq = False,
            scaler=MinMaxScaler()
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
            print(f"ERROR: File missing in {str(LAST_PREPROCESS_FOLDER)}. Now run preprocessing...")
            return main(seq_len, do_preprocess)
        

    dim_ffn = 126
    d_model = 32
    activation_dec: Union[str | Callable[[torch.Tensor], torch.Tensor]] = nn.SiLU()

    transformer_decoder_params = {
        "d_model": d_model,
        "nhead": 8,
        # "num_encoder_layers": 6,
        # "num_decoder_layers": 2,
        "dim_feedforward": dim_ffn,
        "dropout": 0.1,
        "activation": activation_dec,
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
            self.ffn = nn.Linear(d_model, num_outputs)
            self.act_out = act_out # nn.Sigmoid()
            
        def forward(self, x):
            len_b, len_s, _ = x.shape
            emb = self.emb_layer(x)
            out = self.model(emb, emb)
            out = out[:, -1, :].view(len_b, 1, -1)
            
            if self.act_out:
                return self.act_out(self.ffn(out))
            return self.ffn(out)

    model = DecoderModel(act_out=nn.Sigmoid())

    trainer = Trainer(
        model=model,
        loss=nn.MSELoss(),
        optimizer=torch.optim.AdamW(params=model.parameters()),
        device=DEVICE,
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
        epochs=700,
        eval_on_test=True,
        k_folds=0,
    )

    score = trainer.eval(X_val, y_val)
    import numpy as np

    try:
        print("Score on validation set (rmse):", np.sqrt(score))
    except:
        print("Score on validation set (rmse):", np.sqrt(score.cpu().numpy()))

    
    # PREDICTION STEP

    grouped_test = test_set.groupby("vesselId")

    predictions = []

    sequence_length = seq_len

    for vessel_id, group in tqdm(grouped_test, colour="green"):
        forecast_steps = len(group['time'].values) - seq_len

        last_known_features = group[features_input].values

        future_preds = iterative_forecast(last_known_features, trainer, forecast_steps, sequence_length)
        
        group.loc[group.index[seq_len:],['cog', 'sog', 'rot', 'heading', 'latitude', 'longitude']] = future_preds
        
        group[features_input] = scaler.inverse_transform(group[features_input])
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

    submit(res)

if __name__ == "__main__":
    seq_len = 48
    do_preprocess = True
    main(seq_len, do_preprocess)