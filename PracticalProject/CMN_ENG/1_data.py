# %%

import torch
from torchtext.legacy.data import get_tokenizer
from torchtext.legacy.vocab import Vocab, build_vocab_from_iterator
from torch.utils.data import Dataset
import unicodedata
import opencc
import os

from config import *

# %%

"""
定义超参
"""
# MAX_LENGTH = 20
# PROJECT_ROOT_PATH = os.path.abspath('.') + os.path.sep + os.path.join('..', '..')

# %%

"""
简体 -> 繁体
"""
cc = opencc.OpenCC('t2s')

# %%

"""
将 unicode -> ASCII
"""


def unicodeToAscii(s: str):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# %%

"""
过滤长句
"""


def filterPair(p):
    return len(p[0]) < MAX_LENGTH and len(p[1]) < MAX_LENGTH


# %%

"""
分词
"""

token_transform_en = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
token_transform_zh = get_tokenizer(tokenizer='spacy', language='zh_core_web_sm')


# %%

class MyData(Dataset):
    def __init__(self, path):
        self.pairs = []
        for line in open(path, encoding='utf-8'):
            pair = cc.convert(line).split('\t')
            pair[0] = token_transform_en(unicodeToAscii(pair[0].lower().strip()))
            pair[1] = token_transform_zh(unicodeToAscii(pair[1].lower().strip()))
            pair.pop()
            if filterPair(pair):
                self.pairs.append(pair)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1]

    def __len__(self):
        return len(self.pairs)


# MyData(path=os.path.join(PROJECT_ROOT_PATH, 'data', 'cmn-eng', 'cmn.txt'))

# %%

"""
token 迭代器
"""


def yield_tokens(data_iter, id: int):
    for data_sample in data_iter:
        yield data_sample[id]


# %%

"""
vocab 生成
"""


def vocab_generate():
    data = MyData(path=os.path.join(PROJECT_ROOT_PATH, 'data', 'cmn-eng', 'cmn.txt'))
    eng = build_vocab_from_iterator(yield_tokens(data_iter=data, id=0), )
    zh = build_vocab_from_iterator(yield_tokens(data_iter=data, id=1), )
    return data, eng, zh


# %%

"""
实现vocab
"""
data, eng, zh = vocab_generate()
# CMN_ENG_PATH = os.path.join(PROJECT_ROOT_PATH, 'Example', 'CMN-ENG')
torch.save(eng, os.path.join(vocab_path, "en_vocab"))
torch.save(zh, os.path.join(vocab_path, "zh_vocab"))


# %%

def sentence2index(sentence: list, vocab, MAX_LENGTH) -> (list, list):
    index = []
    if len(sentence) < MAX_LENGTH:
        valid_len = [len(sentence)]
        sentence += ['<pad>'] * (MAX_LENGTH - len(sentence))
    else:
        valid_len = [MAX_LENGTH]
        sentence = sentence[:MAX_LENGTH]
    for token in sentence:
        if token not in vocab.stoi:
            token = '<unk>'
        index.append(vocab.stoi[token])
    return index, valid_len


sentence = ['hi', '.']
vocab = eng
sentence2index(sentence, vocab, 20)


# %%

# eng.
# eng.stoi
# eng.itos

# %%


# 生成语料
def load_corpus(data, eng_vocab, zh_vocab, MAX_LENGTH=MAX_LENGTH):
    sentence = []
    valid_len = []
    for en_lang, zh_lang in data.pairs:
        en_index, en_valid_len = sentence2index(en_lang, eng_vocab, MAX_LENGTH)
        zh_index, zh_valid_len = sentence2index(zh_lang, zh_vocab, MAX_LENGTH)
        sentence.append([en_index, zh_index])
        valid_len.append([en_valid_len, zh_valid_len])
    return sentence, valid_len


# %%

print(corpus_path)

sentence, valid_len = load_corpus(data, eng, zh)

print(sentence[:3], valid_len[:3])
sentence = torch.tensor(sentence, dtype=torch.long)
valid_len = torch.tensor(valid_len, dtype=torch.long)
torch.save(sentence, os.path.join(corpus_path, 'sentence'))
torch.save(valid_len, os.path.join(corpus_path, 'valid_len'))
print(sentence.shape)
print(valid_len.shape)
