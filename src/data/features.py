import pandas as pd
from dataclasses import dataclass
import numpy as np
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
        .reset_index(level=0, drop=True)
        .reindex(df.index)
    )#.fillna(0)  # Fill remaining NaNs with a default value, e.g., 0

    df_one_hot_encoded = pd.get_dummies(df[column], prefix=column, drop_first=False, dtype='float32')

    df = df.join(df_one_hot_encoded)
    return df

def create_heading_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    CREATE `heading_cos` AND `heading_sin` FEATURES

    Args:
        - df: pd.DataFrame = ais_train and ais_test concatenated (ais_data)
    Returns:
        - ais_data: pd.DataFrame = ais_data with `heading_cos` and `heading_sin` features
    """
    
    # Replace 511 (not available) with NaN
    df['heading'] = df['heading'].replace(511, pd.NA)
    
    # Calculate the cosine and sine of the heading
    df['heading_cos'] = df['heading'].apply(lambda x: 0 if pd.isna(x) else np.cos(np.deg2rad(x)))
    df['heading_sin'] = df['heading'].apply(lambda x: 0 if pd.isna(x) else np.sin(np.deg2rad(x)))
    df.drop(columns=['heading'], inplace=True)
    return df

def create_rot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    CREATE `rot_calculated` AND `rot_category` FEATURES

    Args:
        - df: pd.DataFrame = ais_train and ais_test concatenated (ais_data)
    Returns:
        - ais_data: pd.DataFrame = ais_data with `rot_calculated` and `rot_category` features
    """
    
    # Calculate the ROT in degrees per minute
    df['rot_calculated'] = 4.733 * (df['rot'].abs() ** 0.5) * df['rot'].apply(lambda x: 1 if x >= 0 else -1)
    
    # Categorize the ROT values
    def categorize_rot(rot):
        if rot == -128 or rot == 128:
            return 'no_info'
        elif rot == 127:
            return 'right_more_than_5_deg_per_30s'
        elif rot == -127:
            return 'left_more_than_5_deg_per_30s'
        elif rot > 0:
            return 'turning_right'
        elif rot < 0:
            return 'turning_left'
        else:
            return 'no_turn'
    
    df['rot_category'] = df['rot'].apply(categorize_rot)
    df = one_hot_encode(df, 'rot_category')
    return df

def sog_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    CREATE `sog` FEATURE

    Args:
        - df: pd.DataFrame = ais_train and ais_test concatenated (ais_data)
    Returns:
        - ais_data: pd.DataFrame = ais_data with `sog` feature preprocessed
    """
    
    # Replace 1023 (not available) with NaN and 1022 with 102.2 knots
    df['sog'] = df['sog'].replace(1023, pd.NA).replace(1022, 102.2)
    
    # Convert SOG to knots (1/10 knot steps)
    df['sog'] = df['sog'] / 10.0

    # Replace NaN values with the mean value of the vessel
    df['sog'] = (
        df
        .groupby('vesselId')['sog']
        .transform(lambda x: x.fillna(x.mean()))
    )
    
    # If the value is still NaN, replace it with the mean value of all vessels
    df['sog'] = df['sog'].fillna(df['sog'].mean())
    
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
