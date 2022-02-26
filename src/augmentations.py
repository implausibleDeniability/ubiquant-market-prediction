import pandas as pd


def mixup(x: pd.DataFrame, y: pd.Series, n_samples: int=3000000) -> tuple:
    """Generate new samples from existing data
    x_new_sample = 0.9 * x_base + 0.1 * x_noise
    y_new_sample = y_base
    """
    x_base = x.sample(n=n_samples, replace=True)
    x_noise = x.sample(n=n_samples, replace=True)
    x_new_sample = x_base
    x_new_sample.loc[:, 'f_0':] *= 0.9
    x_new_sample.loc[:, 'f_0':] += 0.1 * x_noise.loc[:, 'f_0':].values
    y_new_sample = y.loc[x_base.index]
    return x_new_sample, y_new_sample