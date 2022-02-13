import datetime
import os
from pathlib import Path

import torch


class WeightsLogger:
    def __init__(self, weights_dir: str, exp_name: str):
        self.weights_dir = Path(weights_dir) / exp_name
        os.makedirs(self.weights_dir, exist_ok=True)

    def save(self, model_weights) -> None:
        """
        Args:
            model_weights: the guy that model.state_dict() returns
        """
        filename = "checkpoint " + str(datetime.datetime.now()) + ".pth"
        path = self.weights_dir / filename
        torch.save(model_weights, path)
