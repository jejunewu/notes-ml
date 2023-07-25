import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, in_put, out_put):
        super(Net, self).__init__()

        self.fc = nn.Linear(in_features=in_put, out_features=out_put)

    def forward(self, x):
        y = self.fc(x)
        return y


if __name__ == '__main__':
    net = Net(20, 100)
    x = torch.randn([10, 20])
    y = net(x)
    print(y.shape)
