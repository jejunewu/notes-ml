import torch
from torch import nn
from torch.nn import functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        x, hidden = self.gru(embedded, hidden)
        return x, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


from ml_utils.torchwu import torchwu

model = EncoderRNN(vocab_size=1000, input_size=6, hidden_size=3).to(DEVICE)
encoder_hidden = model.initHidden()
x = torch.randint(1, 100, size=(98, 20), device=DEVICE)
print(x.shape)
y = model.forward(x, hidden=encoder_hidden)
print(y.shape)


# model.summary(input_shape=(20,), input_dtype=torch.long)


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        x = self.embedding(input).view(1, 1, -1)
        x = F.relu(x)
        x, hidden = self.gru(x, hidden)
        x = self.fc(x)
        x = self.softmax(x)
        return x, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
