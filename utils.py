from settings import *

from datetime import datetime
import uuid

import pandas as pd

def make_file_name() -> str:
    file_name = str(uuid.uuid4())
    print(f"Submission file name is: {file_name}")
    return file_name


def submit(forecast: pd.DataFrame, file_name: str = None) -> None:
    sample_submission = pd.read_csv(AIS_SAMPLE_SUBMISSION)
    file_name = file_name if file_name else make_file_name()

    repertory = DATA_FOLDER.joinpath(file_name)
    sample_submission = sample_submission[['id']].merge(forecast[["ID","longitude_predicted","latitude_predicted"]], on='id', how='left')
    try:
        sample_submission.to_csv(repertory, index=False)
    except:
        print("Error register file")
        submit(forecast)
