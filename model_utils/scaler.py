import numpy as np
import math
import torch
import torch.nn as nn

def get_scaler(scaling_type):
    if scaling_type == "std":
        return LogStandardScaler()
    elif scaling_type == "mad":
        pass
        # return LogMADScaler()
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