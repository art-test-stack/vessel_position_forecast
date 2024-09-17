from pathlib import Path

DATA_FOLDER = Path("data/")

AIS_TEST = DATA_FOLDER.joinpath("ais_test.csv")
AIS_TRAIN = DATA_FOLDER.joinpath("ais_train.csv")
PORTS = DATA_FOLDER.joinpath("ports.csv")
SCHEDULES_TO_MAY_2024 = DATA_FOLDER.joinpath("schedules_to_may_2024.csv")
VESSELS = DATA_FOLDER.joinpath("vessels.csv")
