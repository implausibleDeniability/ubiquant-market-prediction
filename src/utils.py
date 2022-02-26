import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


def make_sample():
    load_dotenv()

    dataset_dir = Path(os.environ["dataset_dir"])
    data = pd.read_csv(dataset_dir / "train.csv")
    data.sample(frac=0.01).to_csv(dataset_dir / "train_sample.csv")

    
def convert2feather():
    load_dotenv()
    
    dataset_dir = Path(os.environ["dataset_dir"])
    data = pd.read_csv(dataset_dir / "train.csv")
    data.to_feather(dataset_dir / 'train.feather')
