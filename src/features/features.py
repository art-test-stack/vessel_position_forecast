import pandas as pd
from dataclasses import dataclass

from typing import Union


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
    # 'navstat',
    # 'etaRaw',
    'latitude',
    'longitude',
]

# TODO: 
#   - add season columns?
#   - add hour columns?
#   - change timestamp to sin?
#   - pick location data? (from other datasets)
#   - add time diff with arrival (etaRaw - time)

def create_time_diff_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    CREATE time_diff AND MAKE IT IN SECONDS
    
    input:
        - ais_data: pd.DataFrame = ais_train and ais_test concatenated
    """

    df['time_diff'] = (
        df
        .sort_values(by=['time'])
        .groupby("vesselId")['time']
        .diff()
        .dropna()
        .dt.total_seconds()
        .astype(int)
        .shift(-1)
    )

    # arrival time diff (from etaRaw)
    # 
    return df


def presequence_data(df: pd.DataFrame, seq_len: int = 1) -> pd.DataFrame:

    def update_split_column(group: pd.Series) -> pd.Series:
        if group.name in df["vesselId"].unique():
            group.iloc[-seq_len] = "both"
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

