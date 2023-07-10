import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Net, self).__init__()
        self.linear = nn.Linear(in_features=in_channel, out_features=out_channel, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear(x)
        y = self.sigmoid(y)
        return y


in_channel = 3
out_channel = 2
x = torch.randn(size=(1, 1, 1, in_channel))
model = Net(in_channel, out_channel)
print(model)
y = model(x)
print(x.shape)
print(y.shape)
