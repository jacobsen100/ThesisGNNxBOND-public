import torch


def MAPE(y, y_hat):
    """Mean Absolute Percentage Error"""
    return torch.abs(y_hat - y) / torch.abs(y)


def MAE(y, y_hat):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(y_hat - y))


def MSE(y, y_hat):
    """Mean Square Error"""
    return torch.mean((y_hat - y) ** 2)


def RMSE(y, y_hat):
    """Root Mean Square Error"""
    return torch.sqrt(torch.mean((y_hat - y) ** 2))


def SE(y, y_hat):
    """Squared Error"""
    return torch.sum((y_hat - y) ** 2)


def RS(y, y_mean):
    return torch.sum((y - y_mean) ** 2)


def AE(y, y_hat):
    """Absolute Error"""
    return torch.sum(torch.abs(y - y_hat))


def RA(y, y_mean):
    return torch.sum(torch.abs(y - y_mean))
