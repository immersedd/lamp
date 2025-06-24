import numpy as np
import math
import torch
import torch.nn as nn

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
    sigma = torch.exp(log_var) + 1e-6  # 防止除以零，保证方差不会太小

    # 计算高斯损失
    loss = 0.5 * torch.log(sigma) + 0.5 * torch.div(torch.pow(y_true - mu, 2), sigma)
    # 求平均损失
    return torch.mean(loss)