import datetime
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.data import Dataset
from src.metrics import pearson_metric
from src.torch_models import EmbedMLP
from src.weights_logger import WeightsLogger

# configs
load_dotenv()


@click.command()
@click.option("--exp-name", type=str, default="unnamed_run", help="Name of experiment")
@click.option("--config", type=str, default="base", help="Training configuration")
@click.option("--debug", is_flag=True, default=False)
@click.option("--cpu", is_flag=True, default=False)
def main(exp_name: str, config: str, debug: bool, cpu: bool):
    wandb_mode = "disabled" if debug else "online"
    wandb.init(
        project="market_prediction", entity="parmezano", name=exp_name, mode=wandb_mode
    )
    with open(f"configs/{config}.yaml", "r") as stream:
        wandb.config = yaml.safe_load(stream)
    wandb.config["device"] = "cpu" if cpu else "cuda:0"

    data = load_data(debug=debug)
    train, test = make_splits(data)
    train, test = process_investment_id(train, test)
    train, test = train.drop("time_id", axis=1), test.drop("time_id", axis=1)

    train_dataloader = DataLoader(Dataset(train), batch_size=wandb.config["batch_size"])
    test_dataloader = DataLoader(Dataset(test), batch_size=wandb.config["batch_size"])
    model = EmbedMLP(
        input_dim=301, num_embeddings=3775, dropout=wandb.config["dropout"]
    )
    model = model.to(wandb.config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=wandb.config["epochs"],
        steps_per_epoch=len(train_dataloader),
    )
    weights_logger = WeightsLogger(os.environ["weights_dir"], exp_name, debug)
    train_model(
        model, train_dataloader, test_dataloader, optimizer, scheduler, weights_logger
    )


def load_data(debug=False) -> pd.DataFrame:
    dataset_dir = Path(os.environ["dataset_dir"])
    if debug:
        data = pd.read_csv(dataset_dir / "train_sample.csv", index_col=0)
    else:
        data = pd.read_csv(dataset_dir / "train.csv")
    data = data.set_index("row_id")
    data = data.sample(
        frac=wandb.config["data_sample_frac"]
    )  # because full data doesn't fit into memory
    # investment ids start from 0, but I need the 0th investment to be "unseen" investment
    # that we will use if the new (unseen) investment in the test set arises
    data.investment_id += 1
    return data


def make_splits(data: pd.DataFrame) -> tuple:
    # train on earlier data, test on later data
    train = data.query("time_id < 1000")
    test = data.query("time_id >= 1000")
    return train, test


def process_investment_id(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple:
    learned_investments = train.investment_id.unique()
    new_investments_in_test = test.query(
        "investment_id not in @learned_investments"
    ).index
    test.loc[new_investments_in_test].investment_id = 0
    return train, test


def make_datasets(train: pd.DataFrame, test: pd.DataFrame):
    train_dataset = Dataset(train.drop("time_id", axis=1))
    test_dataset = Dataset(test.drop("time_id", axis=1))
    return train_dataset, test_dataset


def train_epoch(model, dataloader, optimizer, scheduler, criterion) -> list:
    model.train()
    losses = []
    investment_id_dropout = wandb.config["investment_id_dropout"]
    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        x, y_true = batch
        x[:, 0] *= torch.rand(len(x)) > investment_id_dropout
        x = x.to(wandb.config["device"])
        y_true = y_true.to(wandb.config["device"])

        y_pred = model(x)
        loss = criterion(y_true, y_pred.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def eval_epoch_loss(model, dataloader, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x = batch[0].to(wandb.config["device"])
            y_true = batch[1].to(wandb.config["device"])
            y_pred = model(x)
            loss = criterion(y_true, y_pred.view(-1))
            losses.append(loss.item())
    return losses


def train_model(
    model, train_dataloader, test_dataloader, optimizer, scheduler, weights_logger
) -> None:
    criterion = nn.MSELoss()
    for i in range(wandb.config["epochs"]):
        train_losses = train_epoch(
            model, train_dataloader, optimizer, scheduler, criterion
        )
        test_losses = eval_epoch_loss(model, test_dataloader, criterion)
        weights_logger.save(model.state_dict())
        wandb.log(
            {
                "train_loss": np.mean(train_losses, axis=0),
                "test_loss": np.mean(test_losses, axis=0),
            }
        )


if __name__ == "__main__":
    main()
