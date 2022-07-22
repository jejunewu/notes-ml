import numpy as np
import pandas as pd
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchkeras
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

df = pd.read_csv('Data/covid.csv', index_col=0)
print(df.columns)
df.columns = [0, 1, 2]
df.index = pd.to_datetime(df.index)

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df.index,y=df[0].values)),
# fig.add_trace(go.Scatter(x=df.index,y=df[1].values))
# fig.add_trace(go.Scatter(x=df.index,y=df[2].values))
# fig.show()

df_diff = df.diff(periods=1).dropna()
df_diff.reset_index(drop=True, inplace=True)
# print(df_diff)
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df_diff.index,y=df_diff[0].values)),
# fig.add_trace(go.Scatter(x=df_diff.index,y=df_diff[1].values))
# fig.add_trace(go.Scatter(x=df_diff.index,y=df_diff[2].values))
# fig.show()


WINDOW_SIZE = 8


class Covid19Dataset(Dataset):
    def __len__(self):
        return len(df_diff) - WINDOW_SIZE

    def __getitem__(self, i):
        x = df_diff.loc[i:i + WINDOW_SIZE - 1, :]
        feature = torch.tensor(x.values, dtype=torch.float)
        y = df_diff.loc[i + WINDOW_SIZE, :]
        label = torch.tensor(y.values, dtype=torch.float)
        return (feature, label)


ds_train = Covid19Dataset()
dl_train = DataLoader(ds_train, batch_size=5)

torch.random.seed()


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()

    def forward(self, x, x_input):
        x_out = torch.max((1 + x) * x_input[:, -1, :], torch.tensor(0.0))
        return x_out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3层lstm
        self.lstm = nn.LSTM(input_size=3, hidden_size=4, num_layers=5, batch_first=True)
        self.linear = nn.Linear(4, 3)
        self.block = Block()

    def forward(self, x_input):
        # print(x_input.shape)
        # print(x_input)
        x = self.lstm(x_input)[0][:, -1, :]
        # x = self.lstm(x_input)
        # print(x.shape)
        x = self.linear(x)
        y = self.block(x, x_input)
        return y


model = Net()
model_style = 1
Optim = torch.optim.Adam(model.parameters(), lr=1e-2)


def mspe(y_pred, y_true):
    err_percent = (y_true - y_pred) ** 2 / (torch.max(y_true ** 2, torch.tensor(1e-7)))
    return torch.mean(err_percent)


loss_func = nn.MSELoss()

if model_style == 1:
    model = torchkeras.Model(model)

    # model_info = model.summary(input_shape=(6, 3), input_dtype=torch.FloatTensor)
    # 训练用
    model.compile(loss_func=mspe, optimizer=Optim)
    dfhistory = model.fit(50, dl_train, log_step_freq=10)

else:
    def train_step(model, features, labels):
        Optimizer = Optim
        # 正向传播求损失
        predictions = model.forward(features)
        loss = mspe(predictions, labels)
        # 反向传播求梯度
        loss.backward()
        # 参数更新
        Optimizer.step()
        Optimizer.zero_grad()
        return loss.item()


    def train_model(model, epochs):
        for epoch in range(1, epochs + 1):
            list_loss = []
            for features, labels in dl_train:
                lossi = train_step(model, features, labels)
                list_loss.append(lossi)
            loss = np.mean(list_loss)
            if epoch % 10 == 0:
                print('epoch={} | loss={} '.format(epoch, loss))


    train_model(model, 50)

# 可视化
plt.figure()


