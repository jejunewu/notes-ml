import torch

from tools import Net

if __name__ == '__main__':
    net = Net(20, 100)
    x = torch.randn([10, 20])
    y = net(x)
    print(y.shape)
