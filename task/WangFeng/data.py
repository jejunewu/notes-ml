#%%
import os
import jieba
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from ml_utils.torchwu import torchwu, cuda, metrics

#%%
seq_length = 100  # 句子长度
BATCH_SIZE = 50
EPOCHS = 1

#%%
PROJECT_ROOT_PATH = os.path.join(os.path.abspath('.'), '..', '..', '..')

# 使用jieba进行分词
f = open('new_wangfeng.txt', 'r', encoding='utf-8')
all_str = f.read().replace('\n', '').replace(' ', '')  # 去除空格
f.close()
cut_list = jieba.cut(all_str)
seg_list = []  # 分词后的文本数据
for c in cut_list:
    seg_list.append(c)

print(seg_list)
#%%
# 生成one-hot
vocab = sorted(list(set(seg_list)))
word_to_int = dict((w, i) for i, w in enumerate(vocab))
int_to_word = dict((i, w) for i, w in enumerate(vocab))

n_words = len(seg_list)  # 总词量
n_vocab = len(vocab)  # 词表长度
print('总词汇量：', n_words)
print('词表长度：', n_vocab)

#%%


dataX = []
dataY = []
for i in range(0, n_words - seq_length, 1):
    seq_in = seg_list[i:i + seq_length + 1]
    dataX.append([word_to_int[word] for word in seq_in])
# 乱序
np.random.shuffle(dataX)
for i in range(len(dataX)):
    dataY.append([dataX[i][seq_length]])
    dataX[i] = dataX[i][:seq_length]

n_simples = len(dataX)
print('样本数：', n_simples)
print(dataX[:10])
print(dataY[:10])

#%%
X = torch.tensor(dataX, dtype=torch.float).reshape((-1, seq_length, 1))
Y = torch.tensor(dataY, dtype=torch.float).reshape((-1, 1))
print('Toatal: ', X.shape, '-->', Y.shape)

ds = TensorDataset(X, Y)
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
x, y = next(iter(dl))
print('1Batch: ', x.shape, '-->', y.shape)
#%%
# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=6, num_layers=3, batch_first=True)
        self.fc = nn.Linear(in_features=6, out_features=1)

    def forward(self, x):
        # x is input, size (batch_size, seq_len, input_size)
        x, _ = self.lstm(x)
        # x is output, size (batch_size, seq_len, hidden_size)
        x = x[:, -1, :]
        x = self.fc(x)
        x = x.view(-1, 1, 1)
        return x
#%%
DEVICE = cuda.try_gpu()
model = torchwu.Model(Net())
model.summary(input_shape=(seq_length, 1))
model.compile(
    loss_func=F.mse_loss,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
    device=DEVICE,
    metrics_dict={'R2':metrics.r2_score}
)
dfhistory = model.fit(
    epochs=EPOCHS,
    dl_train=dl,
    log_step_freq=200
)