from src.data.features import (
    create_time_diff_feature, 
    presequence_data, 
    create_long_lat_diff_feature, 
    one_hot_encode, 
    create_heading_features, 
    sog_feature,
    create_rot_features
)
from utils import DATA_FOLDER, LAST_PREPROCESS_FOLDER

import concurrent.futures

import torch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
from uuid import uuid4
from typing import Any, Union, List, Tuple
from tqdm import tqdm
from pathlib import Path
import joblib

features_raw_ais_data = [
    "vesselId", 
    "time",
    'cog',
    'sog',
    'rot',
    'heading',
    'navstat',
    'latitude',
    'longitude',
]

features_input = [
    'time_diff',
    # 'navstat',
    'navstat_1.0', 
    'navstat_2.0', 
    # 'navstat_3.0', 
    # 'navstat_4.0',
    'navstat_5.0', 
    'navstat_6.0', 
    # 'navstat_7.0', 
    'navstat_8.0',
    # 'navstat_9.0', 
    # 'navstat_11.0', 
    # 'navstat_12.0', 
    # 'navstat_13.0',
    # 'navstat_14.0', 
    'navstat_15.0',
    'cog',
    'sog',
    # 'rot',
    'rot_calculated',
    # 'rot_category',
    # 'rot_category_left_more_than_5_deg_per_30s',
    # 'rot_category_no_turn',
    # 'rot_category_right_more_than_5_deg_per_30s',
    # 'rot_category_turning_left',
    # 'rot_category_turning_right',
    # 'heading',
    'heading_cos',
    'heading_sin',
    # TODO: handle lat/long
    # 'latitude',
    # 'longitude',
    # 'long_diff',
    # 'lat_diff',
]

features_to_scale = [
    'time_diff',
    # 'navstat',
    'cog',
    'sog',
    'rot_calculated',
    # 'heading',
    'heading_cos',
    'heading_sin',
    # 'latitude',
    # 'longitude',
    'long_diff',
    'lat_diff',
]


features_output = [
    # 'time_diff',
    'cog',
    'sog',
    'rot_calculated',
    # 'heading',
    'heading_cos',
    'heading_sin',
    'long_diff',
    'lat_diff',
]

features_missing = [
    'cog',
    'sog',
    'rot_calculated',
    'heading_cos',
    'heading_sin',
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
    
    Args:
        - df_train: pd.DataFrame = ais_train
        - df_test: pd.DataFrame = ais_test
        - features: List[str] = list of features needed later in ais_train
    Returns:
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
    
    Args:
        - df: pd.DataFrame = ais_data
    Returns:
        - ais_train: pd.DataFrame = ais_train
        - ais_test: pd.DataFrame = ais_test
    """
    data_train = df[(df["split"]=="train")|(df["split"]=="both")]
    data_test = df[(df["split"]=="test")|(df["split"]=="both")]
    # data_train.dropna(subset="time_diff")

    return data_train, data_test



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
        parallelize: bool = False,
        verbose: bool = False, 
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sequences from ais_train
    
    Args:
        - df_train: pd.DataFrame = ais_train
        - features_in: List of feature names used as input.
        - features_out: List of feature names used as output.
        - seq_len_in: Length of the input sequence.
        - seq_len_out: Length of the output sequence.
        - to_torch: Whether to convert the output to torch Tensors.
    Returns:
        - X: Input sequences.
        - y: Output targets.
    """
    grouped = df_train.sort_values("time").groupby("vesselId")

    X, y = [], []
    dropped_vessel_ids = []
    index = []
    args = [(vessel_id, group, features_in, features_out, seq_len_in, seq_len_out) for vessel_id, group in grouped]

    def _loop(sequences, targets, vessel_id):
        if sequences and targets:
            X.extend(sequences)
            y.extend(targets)
            index.append({"vessel_id": vessel_id, "min_id": len(X), "max_id": len(X) + len(sequences) - 1})
        else:
            dropped_vessel_ids.append(vessel_id)


    if parallelize:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for sequences, targets, vessel_id in tqdm(
                executor.map(_n_in_m_out, args), 
                total=len(args), colour="blue", 
                disable=not verbose
            ):
                _loop(sequences, targets, vessel_id)

    else: 
        for arg in tqdm(args, colour="blue", disable=not verbose):
            sequences, targets, vessel_id = _n_in_m_out(arg)
            _loop(sequences, targets, vessel_id)

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


def plot_correlation_matrix(df: pd.DataFrame, features_input: List[str], features_output: List[str]) -> None:
    correlation_matrix = df[features_input + features_output[-2:]].corr()
    
    file_name = DATA_FOLDER.joinpath(f"corr_{str(uuid4())}.png")
    fig = plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    plt.title('Correlation Matrix Heatmap', size=16)
    
    plt.savefig(file_name)
    plt.close(fig)
    print(f"Correlation matrix saved at: {file_name}")


def preprocess(
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame,
        features_raw: List[str] = features_raw_ais_data,
        seq_type: str = "basic",
        seq_len: int = 1,
        seq_len_out: int | None = 1,
        normalize: bool = True,
        scaler = StandardScaler(),
        verbose: bool = False,
        to_torch: bool = False,
        parallelize_seq: bool = False,
        plot_corr_matrix: bool = False,
        preprocess_folder: Path | str = LAST_PREPROCESS_FOLDER,
    ) -> None:
    """
    PREPROCESS RAW data FROM `*.csv` FILES (only `ais_train.csv` and `ais_test.csv` for now)

    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        features_raw: List of raw features to include
        seq_type: Type of sequence for data processing
        seq_len: Length of input sequence
        seq_len_out: Length of output sequence
        verbose: Whether to print out information during processing
        to_torch: Convert to PyTorch tensors
        normalize: Whether to normalize the data using z-score

    Returns:
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
    df = create_long_lat_diff_feature(df)

    df_lat_long = df_train[["time", "vesselId", "latitude", "longitude"]].copy()
    df = one_hot_encode(df, "navstat")
    
    # UPDATE `split` LABEL IN df
    df = presequence_data(df, test_vessel_ids, seq_len)
    df = create_heading_features(df)
    df = sog_feature(df)
    df = create_rot_features(df)
    
    if plot_corr_matrix:
        try:
            plot_correlation_matrix(df, features_input, features_output)
        except Exception as e:
            print(f"Error while plotting correlation matrix: {e}")

    train_set, test_set = split_train_test_sets(df)

    train_set = train_set.dropna()

    # TODO: Look for better way to normalize
    if normalize: 
        train_set[features_to_scale] = scaler.fit_transform(train_set[features_to_scale])
        test_set[features_to_scale] = scaler.transform(test_set[features_to_scale])
    # train_set = train_set.dropna(subset="time_diff")

    # X = X[features_input]
    # y = y[features_output]
    # MAKE SEQUENCE
    if verbose:
        print(f"Create training sequences on mode '{seq_type}'...")
    if seq_type == "basic":
        X, y, index, dropped_vessel_ids = make_sequences_n_in_m_out(train_set, seq_len_in=1, seq_len_out=1, verbose=verbose, parallelize=parallelize_seq)
        X = X.reshape(-1, X.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        # print(train_set.isna().sum())
        # X = train_set.iloc[:-1][features_input].dropna().values
        # y = train_set.iloc[1:][features_output].dropna().values

        # index = None
        # dropped_vessel_ids = []

    elif seq_type == "n_in_1_out":
        X, y, index, dropped_vessel_ids = make_sequences_n_in_m_out(train_set, seq_len_in=seq_len, seq_len_out=1, verbose=verbose, parallelize=parallelize_seq)

    elif seq_type == "n_in_m_out":
        assert type(seq_len_out) == int, "`seq_len_out` parameter must be an integer (int)"
        X, y, index, dropped_vessel_ids = make_sequences_n_in_m_out(
            train_set, seq_len_in=seq_len, 
            seq_len_out=seq_len_out, 
            verbose=verbose, 
            parallelize=parallelize_seq
        )
    
    print("Split training and validation sets...")
    # TODO: ENHANCE THE `train_test_split` TO GET A VALIDATION SET WHICH MATCH THE REQUIREMENTS OF THE METRIC
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, shuffle=False)

    print("Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)

    X_val = torch.Tensor(X_val)
    y_val = torch.Tensor(y_val)

    torch.save(X_train, preprocess_folder.joinpath("X_train.pt"))
    torch.save(y_train, preprocess_folder.joinpath("y_train.pt"))
    torch.save(X_val, preprocess_folder.joinpath("X_val.pt"))
    torch.save(y_val, preprocess_folder.joinpath("y_val.pt"))

    joblib.dump(scaler, preprocess_folder.joinpath("scaler")) 
    test_set.to_csv(preprocess_folder.joinpath("test_set.csv"))

    df_lat_long.to_csv(preprocess_folder.joinpath("df_lat_long.csv"))

    print(f"Preprocessing ok... Files stored at: {preprocess_folder}")
    print(f"Number of vessels dropped: {len(dropped_vessel_ids)}")
    return X_train, X_val, y_train, y_val, test_set, scaler, df_lat_long, dropped_vessel_ids
