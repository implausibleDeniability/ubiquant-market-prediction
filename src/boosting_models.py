import lightgbm as lgb
import numpy as np
import pandas as pd

from src.metrics import pearson_metric


def evaluate_lightgbm(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.DataFrame,
    model_params: dict,
    categorical_features: list = ["investment_id"],
) -> tuple:
    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=categorical_features,
    )
    bst = lgb.train(model_params, train_data)

    y_pred = bst.predict(x_test)
    y_pred = pd.DataFrame({"target": y_pred}, index=x_test.index)

    rmse = np.sqrt(np.mean((y_test.target - y_pred.target) ** 2))
    pearson_coef = pearson_metric(y_test, y_pred)

    return rmse, pearson_coef
