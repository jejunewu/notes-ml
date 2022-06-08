# %% md
# 1. 读取数据
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
split_ratio = 0.95

df_all = pd.read_csv('data/train_set.csv', sep='\t')
df_all = df_all.head(1000)

n_all = len(df_all)
n_train = int(n_all * split_ratio)
n_valid = n_all - n_train

df_train = df_all.head(n_train)
df_valid = df_all.tail(n_valid)

print('All:{}, Train:{}, Valid:{}'.format(n_all, n_train, n_valid))
df_train.head()
# %%
# 句长分布
token_lens = [len(v['text'].split(' ')) for i, v in df_all.iterrows()]
sns.histplot(token_lens)
plt.xlim([0, 5000])
plt.xlabel('Token count')
plt.show()

token_lens = [v['label'] for i, v in df_all.iterrows()]
sns.histplot(token_lens)
# plt.xlim([0, 20])
plt.xlabel('label')
# %% md
# 2. tokenizer
# %%
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


# %%
def ids2text(df, tokenizer):
    ids_list = df['text'].apply(lambda x: np.array(x.split(' ')).astype(int)).values
    sents_list = tokenizer.batch_decode(ids_list)
    sents_list = [sents.replace(' ', '') for sents in sents_list]
    df['sents'] = sents_list
    return df


df_valid = ids2text(df_valid, tokenizer)
df_train = ids2text(df_train, tokenizer)
df_train.head()
# %% md
# 3. 定义数据集
# %%
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        texts = self.texts[item]
        labels = self.labels[item]

        encoding = tokenizer.encode_plus(
            texts,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'texts': texts,
            'labels': torch.tensor(labels, dtype=torch.long),
            'input_ids': encoding['input_ids'].flatten(),
            # attention_mask: [pad] 的位置是0,其他位置是1
            'attention_mask': encoding['attention_mask'].flatten()
        }


# %%
MAX_LEN = 512
BATCH_SIZE = 8
# %%
ds_train = MyDataset(
    texts=df_train['sents'].to_numpy(),
    labels=df_train['label'].to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

ds_valid = MyDataset(
    texts=df_valid['sents'].to_numpy(),
    labels=df_valid['label'].to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
# %%
dl_train = DataLoader(ds_train, batch_size=8)
dl_valid = DataLoader(ds_valid, batch_size=8)

for i in dl_train:
    print(i)
    break
# %% md
# 4. 构建模型
# %%
from transformers import BertModel

bert_model = BertModel.from_pretrained('bert-base-cased')

"""
bert_model

print(bert_model)

sample_txt = '拉三大纪律；啊发动机打市府恒隆迪塞尔我撒否啥发的，阿瑟我去，爱上大时代沙发。'
print(len(sample_txt))
encoding = tokenizer.encode_plus(
    sample_txt,
    max_length = 512,
    add_special_tokens = True, # [CLS] and [SEP]
    pad_to_max_length = True,
    return_token_type_ids = False,
    return_attention_mask = True,
    return_tensors = 'pt' # pt for pytorch
)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

last_hidden_state, pooled_output = bert_model(
    input_ids = input_ids,
    attention_mask = attention_mask,
    return_dict=False
)
print(last_hidden_state.shape)
print(pooled_output.shape)

"""
# %%
from torch import nn
from transformers import BertModel

bert_model = BertModel.from_pretrained('bert-base-cased')


class TextsClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.dropout(pooled_output)
        output = self.linear(output)
        return self.softmax(output)


model = TextsClassifier(n_classes=14)
for data in dl_train:
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    targets = data['labels']

    print(input_ids.shape)
    print(attention_mask.shape)

    outputs = model(input_ids, attention_mask)

    print(outputs)
    break
# %%
model = TextsClassifier(n_classes=14)

for data in dl_train:
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    targets = data['labels']

    outputs = model(input_ids, attention_mask)

    print(outputs)
    break
# %%
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

EPOCHS = 10

model = TextsClassifier(n_classes=14).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
# loss_fn = nn.CrossEntropyLoss().to(device)

total_steps = len(dl_train) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


# %%
def train_model(model, dl_train, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_preds = 0

    for data in dl_train:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_preds += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(),
                                max_norm=1.0)  # We're avoiding exploding gradients by clipping the gradients of the model using clip_gradnorm.
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_preds.double() / n_examples, np.mean(losses)


# %%
def eval_model(model, dl_valid, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_preds = 0

    with torch.no_grad():
        for data in dl_valid:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_preds += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_preds.double() / n_examples, np.mean(losses)


# %%
from collections import defaultdict

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_model(model, dl_train, loss_fn, optimizer, device, scheduler, len(df_train))
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model, dl_valid, loss_fn, device, len(df_valid))
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print('\n')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        #         torch.save(model.state_dict(),'best_model_state.bin')
        best_accuracy = val_acc

torch.save(model.state_dict(), 'model/bert_model_params.mdl')
# model.load_state_dict(torch.load())