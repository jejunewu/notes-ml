import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10
import warnings

from NNUtils.torchwu import torchwu, cuda, metrics
from notes_model.LeNet import Net32

warnings.filterwarnings("ignore")

SPLIT_RATIO = 0.9
BATCH_SIZE = 200
EPOCHS = 500
DEVICE = cuda.try_gpu()
print(DEVICE)

DATA_HOME = r'D:\\REPO\\Datasets\\cifar-10-python\\'

train_datasets = CIFAR10(DATA_HOME, train=True, download=False)
test_datasets = CIFAR10(DATA_HOME, train=False, download=False)

X = torch.tensor(train_datasets.data).float()
Y = torch.tensor(train_datasets.targets)
X_test = torch.tensor(test_datasets.data).float()
Y_test = torch.tensor(test_datasets.targets)

# 转换为conv输入形状 ：(N, C_{in}, H, W)
X = X.reshape(-1, 3, 32, 32) / 255.
X_test = X_test.reshape(-1, 3, 32, 32)
print('Total: ', X.shape, '-->', Y.shape)
print('Test: ', X_test.shape, '-->', Y.shape)

ds = TensorDataset(X, Y)
ds_test = TensorDataset(X_test, Y_test)

ds_train, ds_val = random_split(ds, [int(len(ds) * SPLIT_RATIO), len(ds) - int(len(ds) * SPLIT_RATIO)])
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, num_workers=0)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=0)

model = torchwu.Model(Net32(in_channels=3, out_features=10))
model.summary(input_shape=(3, 32, 32))
model.compile(
    loss_func=F.cross_entropy,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    device=DEVICE,
    metrics_dict={'acc': metrics.accuracy_multi_clf}
)
dfhistory = model.fit(EPOCHS, dl_train=dl_train, dl_val=dl_val, log_step_freq=200)
print(dfhistory)
