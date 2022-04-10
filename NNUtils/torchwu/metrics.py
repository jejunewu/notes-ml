import torch
from sklearn.metrics import accuracy_score
from torch import nn


def accuracy_multi_clf(y_pred, y_true):
    """
    【多分类准确率指标】
    """
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_true, y_pred_cls)


def accuracy_binary_clf(y_pred, y_true):
    """
    【多分类准确率指标】
    """
    y_pred = torch.where(
        y_pred > 0.5,
        torch.ones_like(y_pred, dtype=torch.float32),
        torch.zeros_like(y_pred, dtype=torch.float32)
    )
    acc = torch.mean(1 - torch.abs(y_true - y_pred))
    return acc


def r2_score(predictions, labels):
    """
    【回归-R2】
    """
    labels_mean = torch.mean(labels)
    SSE = torch.sum(torch.pow((labels - predictions), 2))
    SST = torch.sum(torch.pow((labels - labels_mean), 2))
    r2 = 1 - SSE / SST
    return r2
