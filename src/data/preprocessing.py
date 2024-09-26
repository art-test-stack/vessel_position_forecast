from src.data.features import create_time_diff_feature, presequence_data
import pandas as pd

from typing import Any, Union, List, Tuple


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
    data_train.dropna(subset="time_diff")

    return data_train, data_test


def fit_and_normalize(
        X, 
        y, 
        scaler_x, 
        scaler_y,
    ) -> Union[Any]:
    pass


seq_types = ["basic", "n_in_1_out"]


def preprocess(
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame,
        features_raw: List[str] = features_raw_ais_data,
        seq_type: str = "basic",
        seq_len: int = 1,
        verbose: bool = False
    ) -> None:
    """
    Description:


    Input:
        -
        -
        -
    Output:
        -
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


    # SPLIT DATA
    df = presequence_data(df, test_vessel_ids, seq_len)

    train_set, test_set = split_train_test_sets(df)
    train_set = train_set.dropna(subset="time_diff")


    # MAKE SEQUENCE
    if seq_type == "basic":
        pass

    elif seq_type == "n_in_1_out":
        pass

    