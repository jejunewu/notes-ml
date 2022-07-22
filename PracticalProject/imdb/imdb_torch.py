import torch
from torch import nn
from torch.nn import functional as F
import torchkeras
import string, re
import torchtext
from NNUtils import torchwu

MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的长度
BATCH_SIZE = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
print('DEVICE: ', DEVICE)

# 分词方法
tokenizer = lambda x: re.sub('[%s]' % string.punctuation, "", x).split(" ")


# 过滤掉低频词
def filterLowFreqWords(arr, vocab):
    arr = [[x if x < MAX_WORDS else 0 for x in example]
           for example in arr]
    return arr


TEXT = torchtext.legacy.data.Field(sequential=True, tokenize=tokenizer, lower=True,
                                   fix_length=MAX_LEN, postprocessing=filterLowFreqWords)

LABEL = torchtext.legacy.data.Field(sequential=False, use_vocab=False)

# 2,构建表格型dataset
# torchtext.data.TabularDataset可读取csv,tsv,json等格式
ds_train, ds_test = torchtext.legacy.data.TabularDataset.splits(
    path='D://REPO//Datasets//imdb', train='train.csv', test='test.csv', format='csv',
    fields=[('label', LABEL), ('text', TEXT)], skip_header=True)

# 3,构建词典
TEXT.build_vocab(ds_train)

# 4,构建数据管道迭代器
train_iter, test_iter = torchtext.legacy.data.Iterator.splits(
    (ds_train, ds_test),
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    batch_sizes=(BATCH_SIZE, BATCH_SIZE)
)


class DataLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)

    def __len__(self):
        return self.length

    def __iter__(self):
        # 注意：此处调整features为 batch first，并调整label的shape和dtype
        for batch in self.data_iter:
            yield (torch.transpose(batch.text, 0, 1),
                   torch.unsqueeze(batch.label.float(), dim=1))


dl_train = DataLoader(train_iter)
dl_test = DataLoader(test_iter)

# 查看第一个batch
feature, label = next(iter(dl_train))
print(feature.shape, '-->', label.shape)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量
        self.embedding = nn.Embedding(num_embeddings=MAX_WORDS, embedding_dim=3, padding_idx=1)
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv1d(in_channels=3, out_channels=16, kernel_size=(5,)))
        self.conv.add_module("pool_1", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_1", nn.ReLU())
        self.conv.add_module("conv_2", nn.Conv1d(in_channels=16, out_channels=128, kernel_size=(2,)))
        self.conv.add_module("pool_2", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_2", nn.ReLU())
        self.dense = nn.Sequential()
        self.dense.add_module("flatten", nn.Flatten())
        self.dense.add_module("linear", nn.Linear(6144, 1))
        self.dense.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        y = self.dense(x)
        return y


net = Net()


# 准确率
def accuracy(y_pred, y_true):
    y_pred = torch.where(
        y_pred > 0.5,
        torch.ones_like(y_pred, dtype=torch.float32),
        torch.zeros_like(y_pred, dtype=torch.float32)
    )
    acc = torch.mean(1 - torch.abs(y_true - y_pred))
    return acc


train_mode = 0

if train_mode == 0:

    model = torchwu.Model(net)
    model.summary(input_shape=(200,), input_dtype=torch.LongTensor)
    model.compile(
        loss_func=nn.BCELoss(),
        optimizer=torch.optim.Adagrad(model.parameters(), lr=0.02),
        metrics_dict={"accuracy": accuracy},
        device=DEVICE
    )
    dfhistory = model.fit(20, dl_train, dl_val=dl_test, log_step_freq=200)
    # 评估
    model.evaluate(dl_test)

else:
    model = net.to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


    def train_step(model, features, labels):
        # 梯度清零
        optimizer.zero_grad()
        # 正向求损失
        predictions = model(features)
        loss = F.binary_cross_entropy(predictions, labels)
        metric = accuracy(predictions, labels)
        # 反向求梯度
        loss.backward()
        optimizer.step()
        return loss.item(), metric.item()


    def train_model(model, epochs, dl_train):
        for epoch in range(1, epochs + 1):
            list_loss = []
            list_metric = []
            for features, labels in dl_train:
                features.to(DEVICE)
                labels.to(DEVICE)
                lossi, metrici = train_step(model, features, labels)
                list_loss.append(lossi)
                list_metric.append(metrici)
            loss = torch.mean(torch.tensor(list_loss)).item()
            metric = torch.mean(torch.tensor(list_loss)).item()
            if epoch % 1 == 0:
                print('epoch={} | loss={} | acc={}'.format(epoch, loss, metric))


    # 测试一个batch
    feature, label = next(iter(dl_train))
    loss, metric = train_step(model, feature.to(DEVICE), label.to(DEVICE))
    print('loss={} | acc={}'.format(loss, metric))

    # 训练模型
    train_model(model, 5, dl_train)
