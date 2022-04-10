import collections
import torch
import random
import os
import re

PROJECT_ROOT_PATH = os.path.join(os.path.abspath(__file__), '..', '..', '..')


def read_txt():
    file_path = os.path.join(PROJECT_ROOT_PATH, 'Datasets', 'timemachine.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return [re.sub(r'[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('wrong token：' + token)


def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    def __init__(self,
                 tokens=None,
                 min_freq=0,
                 reserved_token=None,
                 ):
        if tokens is None:
            tokens = []
        if reserved_token is None:
            reserved_token = []

        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # unk
        self.idx_to_token = ['<unk>'] + reserved_token
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def load_corpus_time_machine(max_tokens=-1, token='word'):
    lines = read_txt()
    tokens = tokenize(lines, token)
    vocab = Vocab(tokens)

    corpus = [vocab[token] for token in tokens]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_step):
    # 从第一个时间步序列随机丢弃一些数
    corpus = corpus[random.randint(0, num_step - 1):]
    # 划分子序列数目
    num_subseqs = (len(corpus) - 1) // num_step
    # 按时间步初始化索引
    initial_indices = list(range(0, num_subseqs * num_step, num_step))
    # 将序列随机
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos:pos + num_step]

    num_batches = num_subseqs // batch_size

    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):  # @save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:  # @save
    """加载序列数据的迭代器"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,  # @save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


data_iter, vocab = load_data_time_machine(batch_size=10, num_steps=8)
x,y = next(iter(data_iter))
print(x.sh)
print(y)
# print(data_iter)
