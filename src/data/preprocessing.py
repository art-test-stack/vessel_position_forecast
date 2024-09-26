from src.features import *
import pandas as pd


def concat_train_test_sets(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:

    train_vessel_id_time = df_train[ppfeatures].copy()
    train_vessel_id_time["split"] = "train"
    train_vessel_id_time["ID"] = train_vessel_id_time.index

    test_vessel_id_time = df_test[["ID", "vesselId", "time" ]].copy()
    test_vessel_id_time["split"] = "test"
    df = pd.concat([train_vessel_id_time, test_vessel_id_time], ignore_index=True)

    return df


def preprocess(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    pass