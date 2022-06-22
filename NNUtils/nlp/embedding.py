import gensim
import numpy as np


class LoadWord2VecEmbedding:
    """
    载入大型语料库的word2vec
    Args:
        embedding_path (str): 大型 W2V文件路径
        binary (bool): 是否二进制

    """

    def __init__(self, embedding_path: str, binary=False):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=binary)

    def get_model(self):
        return self.model

    def get_vocab(self) -> dict:
        """
        Returns:
            {token_0 : 0, token_1 : 1, ... }
        """
        return self.model.key_to_index

    def get_vectors(self) -> np.ndarray:
        return self.model.vectors

    def idx2word(self, idx=None) -> str or list:
        if idx is not None:
            return self.model.index_to_key[idx]
        return self.model.index_to_key

    def word2vec(self, word=None) -> dict:
        """
        Returns:
            {token_0 : vec_0, token_1 : vec_1, ... }
        """
        if word and word in self.get_vocab():
            idx = self.get_vocab()[word]
            return {word: self.get_vectors()[idx]}
        return {word: vec for word, vec in zip(self.idx2word(), self.get_vectors())}


class BuildMyW2VFromLargeW2V:
    """
    通过大行W2V载入，提取自己语料的W2W
    """

    def __init__(self, corpus: list, large_embedding_path: str):
        self.corpus = corpus
        self.large_embedding_loader = LoadWord2VecEmbedding(large_embedding_path)
        self.large_w2v = self.large_embedding_loader.get_model()
        self.vector_size = self.large_w2v.vector_size
        self.token2vector = {}
        for token in self.corpus:
            if token in self.large_embedding_loader.get_vocab():
                self.token2vector[token] = self.large_embedding_loader.word2vec(word=token)[token]
            else:
                self.token2vector[token] = np.zeros(self.vector_size)

    def get_vocab(self) -> dict:
        """
        Returns:
            {token_0 : vec_0, token_1 : vec_1, ... }
        """
        vocab = {token: i for i, token in enumerate(self.corpus)}
        return vocab

    def get_vectors(self) -> np.ndarray:
        """
        获取所有词向量
        """
        vectors = np.conj(list(self.token2vector.values()))
        return vectors

    def token2vec(self, token=None):
        """
        查询单个词向量
        """
        if token and token in self.token2vector:
            return {token: self.token2vector[token]}
        else:
            return self.token2vector
