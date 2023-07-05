# %%
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.datasets import MNIST

from NNUtils.torchwu import torchwu, cuda, metrics
from notes_model.LeNet import Net28

SPLIT_RATIO = 0.8
BATCH_SIZE = 100
EPOCHS = 8
DEVICE = cuda.try_gpu()
print(DEVICE)

DATA_HOME = r'D:\\REPO\\Datasets\\'
train_datasets = MNIST(DATA_HOME, train=True, download=False)
test_datasets = MNIST(DATA_HOME, train=False, download=False)

X = train_datasets.data.float()
Y = train_datasets.targets
X_test = test_datasets.data.float()
Y_test = test_datasets.targets

# 转换为conv输入形状 ：(N, C_{in}, H, W)
X = X.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
print('Total: ', X.shape, '-->', Y.shape)
print('Test: ', X_test.shape, '-->', Y_test.shape)

ds = TensorDataset(X, Y)
ds_test = TensorDataset(X_test, Y_test)

ds_train, ds_val = random_split(ds, [int(len(ds) * SPLIT_RATIO), len(ds) - int(len(ds) * SPLIT_RATIO)])
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, num_workers=0)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=0)

model = torchwu.Model(Net28())
# model.summary(input_shape=(1, 28, 28))
model.compile(
    loss_func=F.cross_entropy,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
    device=DEVICE,
    metrics_dict={'acc': metrics.accuracy_multi_clf}
)
dfhistory = model.fit(EPOCHS, dl_train=dl_train, dl_val=dl_val, log_step_freq=200)
print(dfhistory)
