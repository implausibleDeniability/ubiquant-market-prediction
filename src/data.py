import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x = data.drop('target', axis=1).values.astype(np.float32)
        self.y = data['target'].values.astype(np.float32)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self,):
        return len(self.x)