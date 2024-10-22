import collections
import numpy as np
import random
import os

PROJECT_ROOT_PATH = os.path.join(os.path.abspath(__file__), '..', '..', '..')


def read_WangFeng():
    with open(os.path.join(PROJECT_ROOT_PATH, 'Datasets', 'wangfeng.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines


def tokenize(lines, token='word'):
    return [list(line) for line in lines]


def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)


class Vocab:
    def __init__(
            self,
            tokens=None,
            min_freq=0,
            reserved_token=None
    ):
        if tokens is None:
            tokens = []
        if reserved_token is None:
            reserved_token = []

        counter = count_corpus(tokens=tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

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


def load_corpus(max_tokens=-1):
    lines = read_WangFeng()
    tokens = tokenize(lines=lines)
    vocab = Vocab(tokens=tokens)
    corpus = [vocab[token] for token in tokens]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):  # @save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield np.array(X), np.array(Y)


if __name__ == '__main__':
    # tmp = read_WangFeng()
    corpus, vocab = load_corpus()
    x,y = next(iter(seq_data_iter_random(corpus, 1, 20)))
    # tmp = vocab.token_freqs[:10]
    print(x.shape, y.shape)
    print(x)
    print('---')
    print(y)
