from torch import nn
from torch.nn import functional as F


class Net28(nn.Module):
    """
    用于(-1,C_in,28,28)形状卷积网络
    """

    def __init__(self, in_channels=1, out_features=10):
        super(Net28, self).__init__()
        # (batch, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0)
        # (batch, 6, 24, 24)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (batch, 6, 12, 12)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # (batch, 16, 8, 8)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (batch, 16, 4, 4)
        self.flatten = nn.Flatten()
        # (batch, 16*4*4=256)
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        # (batch, 128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        # (batch, 64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        # (batch, 32)
        self.fc4 = nn.Linear(in_features=32, out_features=out_features)
        # (batch, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


class Net32(nn.Module):
    """
    用于(-1,C_in,32,32)形状卷积网络
    """

    def __init__(self, in_channels=3, out_features=10):
        super(Net32, self).__init__()
        # (batch, 3, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0)
        # (batch, 6, 28, 28)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (batch, 6, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # (batch, 16, 10, 10)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (batch, 16, 5, 5)
        self.flatten = nn.Flatten()
        # (batch, 16*5*5=400)
        self.fc1 = nn.Linear(in_features=400, out_features=128)
        # (batch, 128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        # (batch, 64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        # (batch, 32)
        self.fc4 = nn.Linear(in_features=32, out_features=out_features)
        # (batch, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
