import numpy as np

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