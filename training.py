#!/usr/bin/env python
# See the current state of R&D in [Notion](https://www.notion.so/MLP-0f9b2dbf6013405385eac14638a9a587).

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from scipy import stats

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from src.metrics import pearson_metric
from src.torch_models import EmbedMLP, MLP
from src.data import Dataset

load_dotenv()

# configs

device = "cuda"
dataset_dir = Path(os.environ['dataset_dir'])

# hyperparameters
investment_id_dropout = 0.01
lr = 9e-4
epochs = 30

# load data
data = pd.read_csv(dataset_dir / 'train.csv')
data = data.sample(frac=0.01)
data = data.set_index('row_id')
data.investment_id += 1

# prepare train test
train = data[data.time_id < 1000]
test = data.query("1000 <= time_id")
learned_investments = train.investment_id.unique()
new_investments_in_test = test.query("investment_id not in @learned_investments").index
test.loc[new_investments_in_test].investment_id = 0

train_dataset = Dataset(train.drop("time_id", axis=1))
test_dataset = Dataset(test.drop("time_id", axis=1))
model = EmbedMLP(input_dim=301, hidden_dim=90, depth=3, activation=nn.ReLU).to(device)
train_dataloader = DataLoader(train_dataset, batch_size=24500)
test_dataloader = DataLoader(test_dataset, batch_size=24500)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=0.026,
                                                epochs=epochs,
                                                steps_per_epoch=len(train_dataloader)                            
            )
loss_function = nn.MSELoss()
for i in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        x, y_true = batch

        x[:, 0] *= (torch.rand(len(x)) > investment_id_dropout)

        x = x.to(device)
        y_true = y_true.to(device)
        
        y_pred = model(x)
        loss = loss_function(y_true, y_pred.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        scheduler.step()
        # TODO: assess by Pearson coefficient
    print(f"EPOCH {i + 1}. Train loss: {train_loss/len(train)}")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            x, y_true = batch[0].to(device), batch[1].to(device)
            y_pred = model(x)
            loss = loss_function(y_true, y_pred.view(-1))
            test_loss += loss.item()
    print(f"EPOCH {i + 1}. Test loss: {test_loss/len(test)}")
