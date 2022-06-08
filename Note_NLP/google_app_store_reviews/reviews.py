#%% md
# Google Play APP 评分分类
#%%
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
# % matplotlib inline
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
#%% md
# 1. 数据读取
#%%
df = pd.read_csv("D://DataModel//Google_Play_Store//reviews.csv")
df = df.head(1000)
df.head()
#%% md
## 1.1 可视化分析
#%%
sns.countplot(df.score)
plt.xlabel('review score');
#%% md
## 1.2情感按评分分3类
#%%
def sentiment(rating):
    if rating < 2:
        return 0

    if rating == 3:
        return 1

    if rating > 3:
        return 2


df['sentiment'] = df.score.apply(sentiment)
df.head()
#%%
ax = sns.countplot(df.sentiment)
plt.xlabel('reviews')
ax.set_xticklabels(['negative', 'neutral', 'positive'])
#%% md
# 2. 定义Tokenizer 和 Datasets
#%%
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=PRE_TRAINED_MODEL_NAME)
#%% md
## 2.1 tokenizer 试用，特殊token
#%%
# sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
# tokens = tokenizer.tokenize(sample_txt)
# token_ids = tokenizer.encode(sample_txt)
# token_ids
print('[SEP]:',tokenizer.sep_token,tokenizer.sep_token_id)
print('[CLS]:',tokenizer.cls_token,tokenizer.cls_token_id)
print('[PAD]:',tokenizer.pad_token,tokenizer.pad_token_id)
#%% md
## 2.2 显示句子长度分布
#%%
token_lens = []

for text in df.content:
    tokens = tokenizer.encode(text,truncation=True, max_length=128)
    token_lens.append(len(tokens))

sns.distplot(token_lens)
plt.xlim([0, 200])
plt.xlabel('Token count')
#%%
sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
encoding = tokenizer.encode_plus(
    sample_txt,
    max_length = 32,
    add_special_tokens = True, # [CLS] and [SEP]
    pad_to_max_length = True,
    return_token_type_ids = False,
    return_attention_mask = True,
    return_tensors = 'pt' # pt for pytorch
)
print(encoding)
print(tokenizer.decode(encoding['input_ids'][0]))
#%% md
## 2.3 创建 Datasets
#%%
class CustomDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        super().__init__()
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self,item):
        review = self.reviews[item]
        target = self.targets[item]

        encoding = tokenizer.encode_plus(
            review,
            max_length = self.max_len,
            add_special_tokens = True,
            pad_to_max_length = True,
            return_token_type_ids = False,
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        return {
            'review': review,
            'target': torch.tensor(target,dtype=torch.long),
            'input_ids': encoding['input_ids'].flatten(),
            # attention_mask: [pad] 的位置是0,其他位置是1
            'attention_mask': encoding['attention_mask'].flatten()
        }
#%%
df.shape
df = df[df['sentiment'].notna()]
df.isnull().sum()
#%%
MAX_LEN = 160
BATCH_SIZE = 6
RANDOM_SEED = 2002
EPOCHS = 10
#%%
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
#%%
df_train.shape, df_test.shape, df_val.shape
#%%
ds = CustomDataset(
    reviews=df.content.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
#%%
def Data_Loader(df, tokenizer, max_len, batch_size):
    ds = CustomDataset(
        reviews=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        # num_workers=0
    )
#%%
train_data_loader = Data_Loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = Data_Loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = Data_Loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
#%%
# for data in train_data_loader:
#     print(data)
#     break
#
# print(data['input_ids'].shape)
# print(data['attention_mask'].shape)
# print(data['target'].shape)

#%% md
# 3. Bert 预训练模型做分类
#%%
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
#%%
last_hidden_state, pooled_output = bert_model(
    input_ids = encoding['input_ids'],
    attention_mask = encoding['attention_mask'],
    return_dict=False
)
print(last_hidden_state.shape)
print(pooled_output.shape)
#%%
bert_model.config.hidden_size
#%% md
## 3.1 建立分类模型
#%%
class SentimentClassifier(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = False
        )
        output = self.dropout(pooled_output)
        output = self.linear(output)
        return self.softmax(output)
#%%
model = SentimentClassifier(n_classes = 3)
model = model.to(device)
#%%
optimizer = AdamW(model.parameters(),lr=2e-5,correct_bias=False)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)
#%%
def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_preds = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['target'].to(device)

        outputs = model(input_ids,attention_mask)
        _, preds = torch.max(outputs,dim=1)
        loss = loss_fn(outputs,targets)
        correct_preds += torch.sum(preds==targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(),max_norm=1.0) #  We're avoiding exploding gradients by clipping the gradients of the model using clip_gradnorm.
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_preds.double() / n_examples, np.mean(losses)
#%%
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_preds = 0

    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['target'].to(device)

            outputs = model(input_ids,attention_mask)
            _, preds = torch.max(outputs,dim=1)
            loss = loss_fn(outputs,targets)
            correct_preds += torch.sum(preds==targets)
            losses.append(loss.item())

    return correct_preds.double() / n_examples, np.mean(losses)
#%%
"""
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_model(model,train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print('\n')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        #         torch.save(model.state_dict(),'best_model_state.bin')
        best_accuracy = val_acc

torch.save(model.state_dict(),'model_params.mdl')
"""
#%%
model = SentimentClassifier(n_classes = 3)
model.load_state_dict(torch.load('model_params.mdl'))
model.to(device)

#%%
def get_preds(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for data in data_loader:
            reviews = data['review']
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            targets = data['target'].to(device)

            review_texts.extend(reviews)
            real_values.extend(targets)

            outputs = model(input_ids,attention_mask)
            _, preds = torch.max(outputs,dim=1)

            prediction_probs.extend(outputs)
            predictions.extend(preds)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()


    return review_texts, predictions, prediction_probs, real_values

#%%
review_texts, predictions, prediction_probs, real_values = get_preds(model,test_data_loader)
predictions

#%%
class_names = ['negative','neutral','positive']
print(classification_report(real_values, predictions, target_names=['negative','neutral','positive']))
#%%
def show_confusion_matrix(confusion_matrix):
    plt.figure(figsize = (12,10))
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Greens")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment');

cm = confusion_matrix(real_values, predictions)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

#%%
review_text = 'This notebook is very useless.'

encoded_review = tokenizer.encode_plus(
    review_text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
)

input_ids = encoded_review['input_ids'].to(device)
attention_mask = encoded_review['attention_mask'].to(device)

output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)

print(f'Review text: {review_text}')
print(f'Sentiment  : {class_names[prediction]}')