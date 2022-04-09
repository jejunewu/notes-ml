import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST, CIFAR10
import torchkeras

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(in_features=20*5*5, out_features=50)
        self.fc1_drop = nn.Dropout()
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x= F.relu(x)
        x = x.view(-1, 20*5*5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_drop(x)
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        y = F.softmax(x,dim=0)
        return y

if __name__ == '__main__':
    DATA_HOME = r'..//..//..//Datasets//CIFAR10//'
    train_datasets = CIFAR10(DATA_HOME, train=True, download=True)
    test_datasets = CIFAR10(DATA_HOME, train=False, download=True)

    X = torch.tensor(train_datasets.data).float()
    Y = torch.tensor(train_datasets.targets)
    X_test = torch.tensor(test_datasets.data).float()
    Y_test = torch.tensor(test_datasets.targets)

    X = torch.reshape(X, (-1, 3, 32, 32))
    print(X.shape, '-->', Y.shape)
    # print(X_test.shape, '-->', Y_test.shape)

    # 显示图片
    # plt.imshow(X[199,:,:])
    # plt.title(Y[199].item())
    # plt.show()

    ds = TensorDataset(X, Y)
    ds_test = TensorDataset(X_test, Y_test)

    split_ratio = 0.8
    ds_train, ds_val = random_split(ds, [int(len(ds) * split_ratio), len(ds) - int(len(ds) * split_ratio)])
    dl_train = DataLoader(ds_train, batch_size=100, num_workers=4, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=100, num_workers=4)
    dl_test = DataLoader(ds_test, batch_size=100, num_workers=4)

    model = torchkeras.Model(Net())
    model.summary(input_shape=(3, 32, 32))
    model.compile(loss_func=F.cross_entropy, optimizer=torch.optim.Adam(model.parameters(), lr=0.1))
    dfhistory = model.fit(100, dl_train=dl_train, dl_val=dl_val, log_step_freq=200)
