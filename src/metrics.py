import pandas as pd
import numpy as np
from scipy import stats


def pearson_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Computes comptetition metric from two dataframes
    Args:
        y_true: pd.DataFrame with 'row_id' index and columns ['time_id', 'target']
        y_pred: pd.DataFrame with 'row_id' index and columns ['target']
    """
    pearson_coefs = []
    for y_true_timestamp in y_true.groupby('time_id')['target']:
        true_price_timestamp = y_true_timestamp[1]
        pred_price_timestamp = y_pred.target[true_price_timestamp.index]
        pearson_timestamp, _ = stats.pearsonr(true_price_timestamp, pred_price_timestamp)
        pearson_coefs.append(pearson_timestamp)
    metric = np.mean(pearson_coefs, axis=0)
    return metric