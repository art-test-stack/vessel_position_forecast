import pandas as pd
from dataclasses import dataclass

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
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    pass 