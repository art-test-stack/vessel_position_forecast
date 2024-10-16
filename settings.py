from pathlib import Path

import torch
import os

# ----------- DATA -----------

DATA_FOLDER = Path("data/")

AIS_SAMPLE_SUBMISSION = DATA_FOLDER.joinpath("ais_sample_submission.csv")
AIS_TEST = DATA_FOLDER.joinpath("ais_test.csv")
AIS_TRAIN = DATA_FOLDER.joinpath("ais_train.csv")
PORTS = DATA_FOLDER.joinpath("ports.csv")
SCHEDULES_TO_MAY_2024 = DATA_FOLDER.joinpath("schedules_to_may_2024.csv")
VESSELS = DATA_FOLDER.joinpath("vessels.csv")


MODEL_FOLDER = Path("models/")

SUBMISSION_FODLER = Path("submissions/")

LAST_PREPROCESS_FOLDER = DATA_FOLDER.joinpath("last_seq_3/")
# LAST_PREPROCESS_FOLDER = DATA_FOLDER.joinpath("last_preprocess/")
# ----------- PROCESSOR -----------

CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
if MPS_AVAILABLE:
    torch.mps.empty_cache()
    torch.mps.set_per_process_memory_fraction(0.)
DEVICE_NAME = "cuda" if CUDA_AVAILABLE else "mps" if MPS_AVAILABLE else "cpu"
DEVICE = torch.device(DEVICE_NAME)

NUM_THREADS = os.cpu_count() # 16