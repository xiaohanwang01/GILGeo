import torch

def get_mselist(y, y_pred):
    mse = (((y - y_pred) * 100) ** 2).sum(dim=1)
    return mse