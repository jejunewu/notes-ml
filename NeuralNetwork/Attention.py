import math
import torch
from torch import nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    """
    多头注意力
     Args:
         ...

    Inputs:
        ...

    Returns:
        ...

    """

    def __init__(
            self,
            key_size,
            query_size,
            value_size,
            num_hiddens,
            num_heads,
            dropout,
            bias=False,
            **kwargs
    ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout=dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        Args:
            queries (torch.Tensor): shape->(batch_size, qkv_dim, num_hiddens)
            keys (torch.Tensor): shape->(batch_size, qkv_dim, num_hiddens)
            values (torch.Tensor): shape->()
            valid_lens (torch.Tensor): shape->()
        Outputs:
            valid_lens (torch.Tensor)
        """
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            #
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)

        output_concat = self.transpose_output(output, self.num_heads)
        # output_concat shape->(batch_size，查询的个数，num_hiddens)
        return self.W_o(output_concat)

    def transpose_qkv(self, X, num_heads):
        """
        为多注意力头的并行变换形状
        Args:
            X (torch.Tensor): shape->(batch_size, qkv_dim, num_hiddens), `qkv_dim`:`num_of_key-value_pairs`.
            num_heads (int): 头个数.
        Outputs:
            X (torch.Tensor): shape->(batch_size*num_heads, qkv_dim, num_hiddens/num_heads)
        """
        batch_size, qkv_dim, num_hiddens = X.shape
        X = X.reshape(batch_size, qkv_dim, num_heads, -1)
        # shape->(batch_size, qkv_dim, num_heads, num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        # shape->(batch_size, num_heads, qkv_dim, num_hiddens/num_heads)
        # out: shape->(batch_size*num_heads, qkv_dim, num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        """
        逆转transpose_qkv函数的操作
        Args:
            X (torch.Tensor): shape->(batch_size*num_heads, kv_dim, num_hiddens/num_heads)
        Outputs:
            X (torch.Tensor): shape->(batch_size, kv_dim, num_hiddens)
        """
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        # shape->(batch_size, num_heads, kv_dim, num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        # shape->(batch_size, kv_dim, num_hiddens)
        return X

class DotProductAttention(nn.Module):
    """
    缩放点积注意力

    """

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Args:
            queries (torch.Tensor): shape->(batch_size，查询的个数，d)
            keys (torch.Tensor): shape->(batch_size，“键－值”对的个数，d)
            values (torch.Tensor): shape->(batch_size，“键－值”对的个数，值的维度)
            valid_lens (torch.Tensor): shape->(batch_size，)或者(batch_size，查询的个数)

        Outputs:
            values (torch.Tensor): shape->(batch_size，“键－值”对的个数，值的维度)

        """
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class PositionalEncoding(nn.Module):
    """
    自注意力（位置注意力）
    将输入加上位置信息

    Args:
        num_hiddens (int) : 隐藏层数.
        dropout (float) : dropout.
        max_len (int) : default=`1000`.
    Inputs:
        X (torch.Tensor): shape->(batch_size, seq_len, feature_size)
    Outputs:
        X (torch.Tensor): shape->(batch_size, seq_len, feature_size)

    """

    def __init__(self, num_hiddens, dropout: float, max_len=1000):
        super(PositionalEncoding, self).__init__()
        nn.Dropout()
        self.dropout = nn.Dropout(dropout)
        # 建立一个长的pos变量
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class MultiHeadAttention(nn.Module):
    """
    多头注意力

    Args:
        key_size (int):键.
        query_size(int):查询.
        value_size(int):值.
        num_hiddens(int):隐藏层.
        num_heads(int):头数.
        dropout (float): dropout.
        bias=False,

    Inputs:
        queries (torch.Tensor): shape->(batch_size, qkv_dim, num_hiddens)
        keys (torch.Tensor): shape->(batch_size, qkv_dim, num_hiddens)
        values (torch.Tensor): shape->()
        valid_lens (torch.Tensor): shape->()
    Outputs:
        valid_lens (torch.Tensor)

    """

    def __init__(
            self,
            key_size,
            query_size,
            value_size,
            num_hiddens,
            num_heads,
            dropout,
            bias=False,
            **kwargs
    ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout=dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        Args:
            queries (torch.Tensor): shape->(batch_size, qkv_dim, num_hiddens)
            keys (torch.Tensor): shape->(batch_size, qkv_dim, num_hiddens)
            values (torch.Tensor): shape->()
            valid_lens (torch.Tensor): shape->()
        Outputs:
            valid_lens (torch.Tensor)
        """
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            #
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)

        output_concat = self.transpose_output(output, self.num_heads)
        # output_concat shape->(batch_size，查询的个数，num_hiddens)
        return self.W_o(output_concat)

    def transpose_qkv(self, X, num_heads):
        """
        为多注意力头的并行变换形状
        Args:
            X (torch.Tensor): shape->(batch_size, qkv_dim, num_hiddens), `qkv_dim`:`num_of_key-value_pairs`.
            num_heads (int): 头个数.
        Outputs:
            X (torch.Tensor): shape->(batch_size*num_heads, qkv_dim, num_hiddens/num_heads)
        """
        batch_size, qkv_dim, num_hiddens = X.shape
        X = X.reshape(batch_size, qkv_dim, num_heads, -1)
        # shape->(batch_size, qkv_dim, num_heads, num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        # shape->(batch_size, num_heads, qkv_dim, num_hiddens/num_heads)
        # out: shape->(batch_size*num_heads, qkv_dim, num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        """
        逆转transpose_qkv函数的操作
        Args:
            X (torch.Tensor): shape->(batch_size*num_heads, kv_dim, num_hiddens/num_heads)
        Outputs:
            X (torch.Tensor): shape->(batch_size, kv_dim, num_hiddens)
        """
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        # shape->(batch_size, num_heads, kv_dim, num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        # shape->(batch_size, kv_dim, num_hiddens)
        return X


###################
# Attention Utils #
###################
def sequence_mask(X: torch.Tensor, valid_len: torch.Tensor, value=0):
    """
    对 dim=1 的 seq_len进行掩蔽
    X: (batch, seq_len, )
    """
    max_len = X.size(1)
    mask = torch.arange((max_len), dtype=torch.float, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X: torch.Tensor, valid_lens: torch.Tensor):
    # X:3D 张量， valid_lens: 1D或2D张量
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X=X.reshape(-1, shape[-1]), valid_len=valid_lens, value=-1e6)
        return F.softmax(X.reshape(shape), dim=-1)
