from settings import *

from src.data.preprocessing import preprocess

import pandas as pd

def main():
    
    # OPEN NEEDED `*.csv` files
    ais_train = pd.read_csv(AIS_TRAIN, sep='|')
    ais_train['time'] = pd.to_datetime(ais_train['time'])

    ais_test = pd.read_csv(AIS_TEST, sep=",")
    ais_test['time'] = pd.to_datetime(ais_test['time']) 

    preprocess(
        ais_train, 
        ais_test,
        seq_type="n_in_m_out",
        seq_len=10,
        seq_len_out=3,
        verbose=True,
        to_torch=True
    )

if __name__ == "__main__":
    main()