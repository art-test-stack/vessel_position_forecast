import pandas as pd
from dataclasses import dataclass

from typing import Union, List


@dataclass
class DataType:
    raw_type: str
    name: str
    values: list


features = [
    "time_diff",

]

ppfeatures = [
    "vesselId", 
    "time",
    # 'time_diff',
    'cog',
    'sog',
    'rot',
    'heading',
    'navstat',
    # 'etaRaw',
    'longitude',
    'latitude',
]

# TODO: 
#   - add season columns?
#   - add hour columns?
#   - change timestamp to sin?
#   - pick location data? (from other datasets)
#   - add time diff with arrival (etaRaw - time)

def create_time_diff_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    CREATE `time_diff` AND MAKE IT IN SECONDS

    Args:
        - df: pd.DataFrame = ais_train and ais_test concatenated (ais_data)
    Returns:
        - ais_data: pd.DataFrame = ais_data with `time_diff` feature
    """

    df['time_diff'] = (
        df
        .sort_values(by=['time'])
        .groupby("vesselId")['time']
        .diff(-1)
        .abs()
        .dropna()
        .dt.total_seconds()
        .astype(int)
    )
    ## OLD ONE:
    #     (
    #     ais_data
    #     .sort_values(by=['time'])
    #     .groupby("vesselId")['time']
    #     .diff()
    #     .abs()
    #     .dropna()
    #     .dt.total_seconds()
    #     .astype(int)
    #     .shift(-1)
    # )
    return df

def create_long_lat_diff_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    CREATE `long_diff` AND `lat_diff`

    Args:
        - df: pd.DataFrame = ais_train and ais_test concatenated (ais_data)
    Returns:
        - ais_data: pd.DataFrame = ais_data with `long_diff` and `lat_diff` features
    """

    df['long_diff'] = (
        df
        .sort_values(by=['time'])
        .groupby("vesselId")['longitude']
        .diff(-1)
        # .abs()
        .dropna()
    )

    df['lat_diff'] = (
        df
        .sort_values(by=['time'])
        .groupby("vesselId")['latitude']
        .diff(-1)
        # .abs()
        .dropna()
    )
    return df

def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    ONE-HOT ENCODE `column` IN `df`

    Args:
        - df: pd.DataFrame = ais_train and ais_test concatenated (ais_data)
        - column: str = column to one-hot encode
    Returns:
        - ais_data: pd.DataFrame = ais_data with one-hot encoded `column`
    """
    df[column] = (
        df
        .sort_values(by=['time'])
        .groupby("vesselId")[column]
        .apply(lambda x: x.ffill().bfill())
    )
    df = pd.get_dummies(df, prefix=column)
    #             pd.concat(
    #     [df.reset_index(drop=True), pd.get_dummies(df[column].reset_index(drop=True), prefix=column)], 
    #     axis=1
    # )
    return df

def presequence_data(
        df: pd.DataFrame, 
        vessel_ids: List[str],
        seq_len: int = 1
    ) -> pd.DataFrame:
    """
    Description:
        UPDATE `split` LABEL FOR SAMPLES NEEDED FOR BOTH training AND test
    
    Args:
        - df: pd.DataFrame = ais_train and ais_test concatenated (ais_data)
        - seq_len: int = length of the sequences
    Returns:
        - ais_data: pd.DataFrame = ais_data with updated `split` label
    """

    def update_split_column(group: pd.Series) -> pd.Series:
        if group.name in vessel_ids:
            group.iloc[-seq_len:] = "both"
        return group

    df_temp = df.copy()

    ser_temp = pd.Series(
        df_temp[df_temp["split"]=="train"]
        .sort_values(by="time")
        .groupby("vesselId")["split"]
        .apply(update_split_column)
        .reset_index(drop=True)
    )
    ser_temp.index = df_temp[df_temp["split"]=="train"].sort_values(["vesselId", "time"]).index

    df.loc[ser_temp.index, "split"] = ser_temp
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    pass 
