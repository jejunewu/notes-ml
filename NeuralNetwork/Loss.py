import torch
from torch import nn


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    掩蔽Softmax损失函数
    张量指定长度掩蔽，dim=1

    Args:
        pred (torch.Tensor): shape->(batch_size, seq_len, vocab_size)
        label (torch.Tensor): shape->(batch_size, seq_len)
        valid_len (torch.Tensor): shape->(batch_size, )

    """

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = self.sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

    def sequence_mask(self, X: torch.Tensor, valid_len: torch.Tensor, value=0):
        """
        对 dim=1 的 seq_len进行掩蔽
        X: (batch, seq_len, )
        """
        max_len = X.size(1)
        mask = torch.arange((max_len), dtype=torch.float, device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
