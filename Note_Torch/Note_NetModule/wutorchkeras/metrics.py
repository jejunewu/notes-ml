import torch
from sklearn.metrics import accuracy_score
from torch import nn


def accuracy(y_pred,y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true,y_pred_cls)

def r2_score(predictions, labels):
    labels_mean = torch.mean(labels)
    SSE = torch.sum(torch.pow((labels - predictions), 2))
    SST = torch.sum(torch.pow((labels - labels_mean), 2))
    r2 = 1 - SSE / SST
    return r2