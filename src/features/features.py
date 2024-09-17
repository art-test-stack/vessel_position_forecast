import pandas as pd
from dataclasses import dataclass

from typing import Union


@dataclass
class DataType:
    raw_type: str
    name: str
    values: list


raw_features = {
    "time": DataType(
        raw_type="str",
        name="time",
        values=["2024-01-01 00:00:25", "2024-05-07 23:59:08"]
    ),
    "cog": DataType(
        raw_type=float,
        name="Course Over Ground",
        values=[0, 360]
    ),
    "sog": DataType(
        raw_type=float,
        name="Speed Over Ground",
        values=[0, 102.3]
    ),
    "rot": DataType(
        raw_type=int,
        name="Rate of turn",
        values=[-127, 128]
    ),
    "heading": DataType(
        raw_type=int,
        name="Heading",
        values=[0, 359],
    ),
    "navstat": DataType(
        raw_type=int,
        name="Navigational Status",
        values=[0, 15]
    ),
    "etaRaw": DataType(
        raw_type=str,
        name="Estimated Time of Arrival - Raw",
        values=["00-00 00:00", "12-31 23:59"],
    ),
    "latitude": DataType(
        raw_type=float,
        name="Latitude",
        values=[-90., 90.],
    ),
    "longitude": DataType(
        raw_type=float,
        name="Latitude",
        values=[-180., 180]
    ),
    "vesselId": DataType(
        raw_type=str,
        name="vesselId",
        values=[]
    ),
    "portId": DataType(
        raw_type=str,
        name="portId",
        values=[]
    )
}

features = [
    "time_diff",

]

def create_time_diff_feature(ais_train: pd.DataFrame, ais_test: pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
    # CREATE time_diff AND MAKE IT IN SECONDS

    train_vessel_id_time = ais_train[["vesselId", "time"]]
    train_vessel_id_time["split"] = "train"
    train_vessel_id_time["ID"] = train_vessel_id_time.index

    test_vessel_id_time = ais_test[["ID", "vesselId", "time" ]]
    test_vessel_id_time["split"] = "test"
    all_times_vesselId = pd.concat([train_vessel_id_time, test_vessel_id_time], ignore_index=True)

    all_times_vesselId['time_diff'] = all_times_vesselId.sort_values(by=['vesselId', 'time']).groupby("vesselId")['time'].diff().shift(-1)

    # arrival time diff (from etaRaw)
    # all_times_vesselId['arr_time_diff'] = all_times_vesselId.sort_values(by=['vesselId', 'time']).groupby("vesselId")['time'].diff().shift(-1)

    ais_test["time_diff"] = all_times_vesselId[all_times_vesselId["split"]=="test"].sort_values(by="ID").reset_index()["time_diff"]
    ais_train["time_diff"] = all_times_vesselId[all_times_vesselId["split"]=="train"].sort_values(by="ID").reset_index()["time_diff"]

    nb_dt_na_test = ais_test["time_diff"].isna().sum()
    ais_test["time_diff"] = ais_test.sort_values(by=["time_diff"]).iloc[:-nb_dt_na_test]["time_diff"].dt.total_seconds().astype(int)

    nb_dt_na_train = ais_train["time_diff"].isna().sum()
    ais_train["time_diff"] = ais_train.sort_values(by=["time_diff"]).iloc[:-nb_dt_na_train]["time_diff"].dt.total_seconds().astype(int)

    return ais_train, ais_test


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    pass 