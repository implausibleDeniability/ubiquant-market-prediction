import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

dataset_dir = Path(os.environ["dataset_dir"])
data = pd.read_csv(dataset_dir / "train.csv")
data.sample(frac=0.01).to_csv(dataset_dir / "train_sample.csv")
