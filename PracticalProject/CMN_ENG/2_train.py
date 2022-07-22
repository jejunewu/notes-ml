# %%
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import *
from NeuralNetwork.Seq2Seq import Seq2SeqDecoder, Seq2SeqEncoder, EncoderDecoder

"""
读取vocab
"""
en_vocab = torch.load(os.path.join(vocab_path, "en_vocab"))
zh_vocab = torch.load(os.path.join(vocab_path, "zh_vocab"))

"""
读取corpus, 预处理
"""

sentence = torch.load(os.path.join(corpus_path, 'sentence'))
valid_len = torch.load(os.path.join(corpus_path, 'valid_len'))
en_sentence = sentence[:, 0, :].to(DEVICE)
zh_sentence = sentence[:, 1, :].to(DEVICE)
en_valid_len = valid_len[:, 0, :].squeeze().to(DEVICE)
zh_valid_len = valid_len[:, 1, :].squeeze().to(DEVICE)

print('sentence: ', en_sentence.shape, '-->', zh_sentence.shape)
print('valid_len: ', en_valid_len.shape, '-->', zh_valid_len.shape)

ds_train = TensorDataset(en_sentence, zh_sentence, en_valid_len, zh_valid_len)
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE)
print('1 batch:')
x, y, x_len, y_len = next(iter(dl_train))
print('x->y: ', x.shape, '-->', y.shape)
print('x_len->y_len: ', x_len.shape, '-->', y_len.shape)

"""
掩蔽交叉熵损失
"""
from NeuralNetwork.Loss import MaskedSoftmaxCELoss

## 定义模型

encoder = Seq2SeqEncoder(
    vocab_size=len(en_vocab),
    embed_size=EMBED_SIZE,
    num_hiddens=NUM_HIDDENS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)
decoder = Seq2SeqDecoder(
    vocab_size=len(zh_vocab),
    embed_size=EMBED_SIZE,
    num_hiddens=NUM_HIDDENS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)
net = EncoderDecoder(encoder=encoder, decoder=decoder).to(DEVICE)
loss_func = MaskedSoftmaxCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)


def train_step(net, features, labels, features_valid_len, labels_valid_len):
    optimizer.zero_grad()
    # 正向传播求损失
    # X:(batch_size, seq_len)
    predictions, state = net.forward(features, labels, features_valid_len)
    loss = loss_func(predictions, labels, labels_valid_len)
    # 反向传播求梯度
    loss.sum().backward()
    # 参数更新
    optimizer.step()
    return loss.sum().item()


# 训练模型
def train_model(model, epochs):
    for epoch in range(1, epochs + 1):
        list_loss = []
        for features, labels, features_valid_len, labels_valid_len in dl_train:
            lossi = train_step(model, features, labels, features_valid_len, labels_valid_len)
            list_loss.append(lossi)
        loss = np.mean(list_loss)
        if epoch % 1 == 0:
            print('epoch={} | loss={} '.format(epoch, loss))


train_model(model=net, epochs=100)
torch.save(net.state_dict(), os.path.join(model_path, 'seq2seq_params.mdl'))
