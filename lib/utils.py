import numpy as np
import torch

class MaxMinScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def fit(self, data):
        data_o = np.array(data)
        self.max = data_o.max()
        self.min = data_o.min()

    def transform(self, data):
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        return (data - min) / (max - min + 1e-12)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min
    

def get_mselist(y, y_pred, y_range):
    y = y * y_range
    y_pred = y_pred * y_range
    mse = (((y - y_pred) * 100) ** 2).sum(dim=1)
    return mse

def bernoulli_loss(r, att):
    loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()
    return loss


def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))