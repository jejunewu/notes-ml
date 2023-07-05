from torch import nn


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
