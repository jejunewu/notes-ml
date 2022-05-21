import collections


def count_corpus(tokens: list) -> dict:
    """
    统计词频

    Args:
        tokens (list): tokens 的数组，shape->1D\2D

    Returns:
        (dict): {token: freq}
    """
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """
    文本构建词典

    Args:
        tokens (list) : tokens 的数组。
        min_freq (int): 最小过滤频率，用`<unk>`代替。
        reserved_tokens (list): 需要添加的token数组，词典index从1开始，0为<unk>

    """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计词频
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # <unk> 的idx 为 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """
        获取词频长度
        """
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """
        将token数组转化为idx数组
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """
        将idx数组转化为token数组
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """
        获取`<unk> 的index
        """
        return 0

    @property
    def token_freqs(self):
        """
        获取词频的数组->[(token_0, freq_0), ...]
        """
        return self._token_freqs
