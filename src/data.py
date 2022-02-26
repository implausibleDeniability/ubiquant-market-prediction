import os 
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x = data.drop("target", axis=1).to_numpy(dtype=np.float32)
        self.y = data["target"].values.astype(np.float32)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self,):
        return len(self.x)


def load_data(use_feather=True, log=True):
    """Loads ubiquant training dataset
    
    Can use feather format to optimize the loading: ~176.26 s. vs ~9.30 s.
    for .csv vs .feather respectfully.
    
    Args:
        use_feather: flag to use optimized loading
        log: flag to print bulking time
    Returns:
        pandas Dataframe with training data    
    """
    start_execution = time.time()
    csv_file = 'train.csv'
    feather_file = 'train.feather'
    
    load_dotenv()
    dataset_dir = Path(os.environ['dataset_dir'])
    if use_feather:
        data = pd.read_feather(dataset_dir / feather_file)
    else:
        data = pd.read_csv(dataset_dir / csv_file)
    data = data.set_index('row_id')
    data.investment_id += 1
    if log:
        print(f"Loading took {time.time() - start_execution:.2f} seconds")
    return data


def build_torch_dataloaders(data, split_data=True, batch_size=25000):
    """Constructs dataloaders for Ubiquant train dataset.
    
    Args:
        split_data: flag to split the data into train and test (ids > 1000)
        batch_size: batch size 
    Returns:
        train and test dataloaders (if not split_data returns None)
    """
    if split_data:
        train = data[data.time_id < 1000]
        test = data.query("1000 <= time_id")
        learned_investments = train.investment_id.unique()
        new_investments_in_test = test.query("investment_id not in @learned_investments").index
        test.loc[new_investments_in_test].investment_id = 0
        
        trainset = Dataset(train.drop("time_id", axis=1))
        testset = Dataset(test.drop("time_id", axis=1))

        testloader = DataLoader(testset, batch_size=batch_size)        
    else:
        train = data.copy()
        trainset = Dataset(train.drop("time_id", axis=1))
        testloader = None
    trainloader = DataLoader(trainset, batch_size=batch_size)
    return trainloader, testloader
