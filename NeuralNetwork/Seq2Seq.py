import torch
from torch import nn
from torch.nn import functional as F


class Seq2SeqEncoder(nn.Module):
    """
    用于序列到序列学习的循环神经网络编码器
    Inputs: X:(batch_size, seq_len)
    Outputs:
        Y: (batch_size, seq_len, num_hiddens)
        state: (num_layers, batch_size, num_hiddens)
    """

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, batch_first=True, dropout=dropout)

    def forward(self, X, *args):
        # X: (batch_size, seq_len)
        X = self.embedding(X)
        # X: (batch_size, seq_len, embed_size)
        # h_0默认为0初始化
        output, state = self.rnn(X)
        # output: (batch_size, seq_len, num_hiddens)
        # state: (num_layers, batch_size, num_hiddens)
        return output, state


class Seq2SeqDecoder(nn.Module):
    """
    用于序列到序列学习的循环神经网络解码器
    Inputs:
        X: (batch_size, seq_len)
        state: (num_layers, batch_size, num_hiddens)

    Outputs:
        output: (batch_size, num_steps, vocab_size)
        state: (num_layers, batch_size, num_hiddens)

    """

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout, batch_first=True)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        enc_output, enc_state = enc_outputs
        return enc_state

    def forward(self, X, state):
        # X: (batch_size, seq_len)
        X = self.embedding(X)
        # X: (batch_size, seq_len, embed_size)
        # state[-1]: (batch_size, num_hiddens)
        context = state[-1].repeat(X.shape[1], 1, 1)
        # context: (seq_len, batch_size, num_hiddens)
        context = context.permute(1, 0, 2)
        # context: (batch_size, seq_len, num_hiddens)
        # 广播context，使其具有与X相同的num_steps
        X_and_context = torch.cat((X, context), 2)
        # X_and_context (batch_size, seq_len, embed_size + num_hiddens)
        output, state = self.rnn(X_and_context, state)
        # output: (batch_size, seq_len, num_hiddens)
        # state:　(num_layers, batch_size, num_hiddens)
        output = self.dense(output)
        # output: (batch_size, seq_len, vocab_size)
        return output, state


class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

"""

from Example.CMN_ENG.config import *

encoder = Seq2SeqEncoder(
    vocab_size=999,
    embed_size=EMBED_SIZE,
    num_hiddens=NUM_HIDDENS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)
decoder = Seq2SeqDecoder(
    vocab_size=3333,
    embed_size=EMBED_SIZE,
    num_hiddens=NUM_HIDDENS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)
net = EncoderDecoder(encoder=encoder, decoder=decoder)  # .to(DEVICE)
enc_X = torch.randint(100, (BATCH_SIZE, MAX_LENGTH))
enc_valid_len = 10

y, _ = net(enc_X, enc_X)
print('net:', y.shape)
enc_outputs = net.encoder(enc_X, 10)
# print('enc_outputs: ', enc_outputs.shape)

dec_state = net.decoder.init_state(enc_outputs, 10)
print('dec_state: ', dec_state.shape)

dec_output, _ = net.decoder(enc_X, dec_state)
print('dec_output: ', dec_output.shape)
print(enc_X.shape)
print(dec_output.shape)
"""