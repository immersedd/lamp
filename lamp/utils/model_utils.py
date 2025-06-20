import numpy as np
import math
import torch
import torch.nn as nn
################################################################## metrics ##################################################################
def mae(y_true, y_pred):
    '''mean_absolute_error'''
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    '''root_mean_squared_error'''
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def mape(y_true, y_pred):
    '''mean_absolute_percentage_error'''
    return np.mean(np.abs((y_true - y_pred) / y_true))*100

def smape(y_true, y_pred):
    '''sMAPE'''
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def get_loss_function(loss_type):

    if loss_type == 'gs':
        criterion = gaussian_loss
    elif loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return criterion

def gaussian_loss(y_true, y_pred):
    mu, log_var = y_pred[:, 0], y_pred[:, 1]
    sigma = torch.exp(log_var) + 1e-6

    loss = 0.5 * torch.log(sigma) + 0.5 * torch.div(torch.pow(y_true - mu, 2), sigma)
    return torch.mean(loss)


def get_scaler(scaling_type):
    if scaling_type == "std":
        return LogStandardScaler()
    else:
        raise ValueError("Invalid method name.")


class LogStandardScaler(object):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, lis):

        arr = np.log(lis)
        self.mean = np.mean(arr)
        self.std = np.std(arr)

    def transform(self, num):
        if isinstance(num, (np.ndarray, torch.Tensor)):
            if isinstance(num, torch.Tensor):
                num = num.detach().cpu().numpy()
            return (np.log(num) - self.mean) / self.std
        else:
            return (math.log(num) - self.mean) / self.std

    def detransform(self, num):
        if isinstance(num, (np.ndarray, torch.Tensor)):
            if isinstance(num, torch.Tensor):
                num = num.detach().cpu().numpy()
            return np.exp(num * self.std + self.mean)
        else:
            return math.exp(num * self.std + self.mean)