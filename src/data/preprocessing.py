from src.data.features import create_time_diff_feature, presequence_data

import concurrent.futures

import torch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from typing import Any, Union, List, Tuple
from tqdm import tqdm

features_raw_ais_data = [
    "vesselId", 
    "time",
    'cog',
    'sog',
    'rot',
    'heading',
    'latitude',
    'longitude',
]

features_input = [
    'time_diff',
    'cog',
    'sog',
    'rot',
    'heading',
    'latitude',
    'longitude',
]

features_output = [
    # 'time_diff',
    'cog',
    'sog',
    'rot',
    'heading',
    'latitude',
    'longitude',
]

seq_types = ["basic", "n_in_1_out", "n_in_m_out"]


def concat_train_test_sets(
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame, 
        features: List[str] = features_raw_ais_data
    ) -> pd.DataFrame:
    """
    Description:

    CONCATENATE `ais_train` AND `ais_test` AND SELECT FINAL features
    
    Input:
        - df_train: pd.DataFrame = ais_train
        - df_test: pd.DataFrame = ais_test
        - features: List[str] = list of features needed later in ais_train
    Output:
        - ais_data: pd.DataFrame = concatenation of ais_train and ais_test
    """

    train_vessel_id_time = df_train[features].copy()
    train_vessel_id_time["split"] = "train"
    train_vessel_id_time["ID"] = train_vessel_id_time.index

    test_vessel_id_time = df_test[["ID", "vesselId", "time" ]].copy()
    test_vessel_id_time["split"] = "test"
    df = pd.concat([train_vessel_id_time, test_vessel_id_time], ignore_index=True)

    return df


def split_train_test_sets(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Description:
        SPLIT `ais_data` INTO `ais_train` AND `ais_test`
    
    Input:
        - df: pd.DataFrame = ais_data
    Output:
        - ais_train: pd.DataFrame = ais_train
        - ais_test: pd.DataFrame = ais_test
    """
    data_train = df[(df["split"]=="train")|(df["split"]=="both")]
    data_test = df[(df["split"]=="test")|(df["split"]=="both")]
    # data_train.dropna(subset="time_diff")

    return data_train, data_test


# def make_sequences_n_in_1_out(
#         df_train: pd.DataFrame,
#         features_in: List[str] = features_input,
#         features_out: List[str] = features_output,
#         seq_len: int = 1,
#         to_torch: bool = False
#     ) -> Any:
#     """
#     Description:
#         ...
    
#     Input:
#         - df_train: pd.DataFrame = ais_train
#         - ...
#     Output:
#         - ...
#     """

#     grouped = df_train.sort_values("time").groupby("vesselId")

#     def _n_in_1_out(data, sequence_length):
#         sequences = []
#         # targets = []
#         for i in range(len(data) - sequence_length):
#             seq = data[i:i+sequence_length].values
#             # target = data[features_in].iloc[i+sequence_length].values
#             sequences.append(seq)
#             # targets.append(target)
#         return sequences # , targets

#     X, y = [], []

#     for _, group in grouped:
#         X_raw = group[features_in].iloc[:-1]
#         y_raw = group[features_out].iloc[seq_len:]

#         sequences = _n_in_1_out(X_raw, seq_len)
#         # sequences, targets = _n_in_1_out(X_raw, seq_len)
#         X.extend(sequences)
#         y.extend(y_raw)

#     X = torch.Tensor(X) if to_torch else np.array(X)
#     y = torch.Tensor(y) if to_torch else np.array(y)


def _n_in_m_out(args):
    vessel_id, data, features_in, features_out, seq_len_in, seq_len_out = args
    sequences = []
    targets = []
    if len(data) < seq_len_in + seq_len_out:
        return (sequences, targets, vessel_id)
    
    for i in range(len(data) - seq_len_in - seq_len_out):
        seq = data[features_in][i:i+seq_len_in].values
        target = data[features_out].iloc[i+seq_len_in:i+seq_len_in+seq_len_out].values
        sequences.append(seq)
        targets.append(target)
    
    return (sequences, targets, vessel_id)

def make_sequences_n_in_m_out(
        df_train: pd.DataFrame,
        features_in: List[str] = features_input,
        features_out: List[str] = features_output,
        seq_len_in: int = 1,
        seq_len_out: int = 1,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        Generates sequences from ais_train
    
    Input:
        - df_train: pd.DataFrame = ais_train
        - features_in: List of feature names used as input.
        - features_out: List of feature names used as output.
        - seq_len_in: Length of the input sequence.
        - seq_len_out: Length of the output sequence.
        - to_torch: Whether to convert the output to torch Tensors.
    Output:
        - X: Input sequences.
        - y: Output targets.
    """
    grouped = df_train.sort_values("time").groupby("vesselId")

    X, y = [], []
    dropped_vessel_ids = []
    index = []
    args = [(vessel_id, group, features_in, features_out, seq_len_in, seq_len_out) for vessel_id, group in grouped]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for sequences, targets, vessel_id in tqdm(executor.map(_n_in_m_out, args), total=len(args), colour="blue", disable=not verbose):
            if sequences and targets:
                X.extend(sequences)
                y.extend(targets)
                index.append({"vessel_id": vessel_id, "min_id": len(X), "max_id": len(X) + len(sequences) - 1})
            else:
                dropped_vessel_ids.append(vessel_id)

    X = np.array(X)
    y = np.array(y)

    return X, y, index, dropped_vessel_ids


def fit_and_normalize(
        X, 
        y, 
        scaler_x, 
        scaler_y,
    ) -> Union[Any]:
    pass


def preprocess(
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame,
        features_raw: List[str] = features_raw_ais_data,
        seq_type: str = "basic",
        seq_len: int = 1,
        seq_len_out: int | None = 1,
        normalize: bool = True,
        verbose: bool = False,
        to_torch: bool = False
    ) -> None:
    """
    Description:
        PREPROCESS RAW data FROM `*.csv` FILES (only `ais_train.csv` and `ais_test.csv` for now)

    Input:
        df_train: Training dataframe
        df_test: Test dataframe
        features_raw: List of raw features to include
        seq_type: Type of sequence for data processing
        seq_len: Length of input sequence
        seq_len_out: Length of output sequence
        verbose: Whether to print out information during processing
        to_torch: Convert to PyTorch tensors
        normalize: Whether to normalize the data using z-score

    Output:
        Preprocessed data ready for training and testing.
    """
    # PUT asserts HERE
    assert seq_type in seq_types, "This type of sequence is not handdled yet"

    # COPY IN CASE
    df_train, df_test = df_train.copy(), df_test.copy()

    test_vessel_ids = list(df_test["vesselId"].unique())

    # CONCAT train AND test SETS
    if verbose:
        print("Concat train and test sets...")
    df = concat_train_test_sets(df_train, df_test, features_raw)

    # CLEAN DATA (future)
    # - look for options (z-score?)
    # - data augmentation (add samples when time_diff > threshold)

    # CREATE FEATURES (can be automated in the future by passing the functions as args)
    if verbose:
        print("Create features...")
    df = create_time_diff_feature(df)


    # UPDATE `split` LABEL IN df
    df = presequence_data(df, test_vessel_ids, seq_len)

    train_set, test_set = split_train_test_sets(df)

    # TODO: Look for better way to normalize
    if normalize: 
        scaler = StandardScaler()
        train_set[features_input] = scaler.fit_transform(train_set[features_input])
        test_set[features_input] = scaler.fit(test_set[features_input])
    # train_set = train_set.dropna(subset="time_diff")

    # MAKE SEQUENCE
    if verbose:
        print(f"Create training sequences on mode '{seq_type}'...")
    if seq_type == "basic":
        pass

    elif seq_type == "n_in_1_out":
        X, y, index, dropped_vessel_ids = make_sequences_n_in_m_out(train_set, seq_len_in=seq_len, seq_len_out=1, verbose=verbose)

    elif seq_type == "n_in_m_out":
        assert type(seq_len_out) == int, "`seq_len_out` parameter must be an integer (int)"
        X, y, index, dropped_vessel_ids = make_sequences_n_in_m_out(train_set, seq_len_in=seq_len, seq_len_out=seq_len_out, verbose=verbose)
    
    print("Split training and validation sets...")
    # TODO: ENHANCE THE `train_test_split` TO GET A VALIDATION SET WHICH MATCH THE REQUIREMENTS OF THE METRIC
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, shuffle=False)

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    print(X_train.mean())

    return X_train, X_val, y_train, y_val, test_set, scaler