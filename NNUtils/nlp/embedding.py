import pandas as pd
import gensim
import numpy as np


class LoadWord2VecEmbedding:
    """
    载入大型语料库的word2vec
    """

    def __init__(self, embdding_path: str, binary=False):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(embdding_path, binary=binary)

    def get_model(self):
        return self.model

    def get_vocab(self) -> dict:
        return self.model.vocab

    def get_vectors(self) -> np.ndarray:
        return self.model.vectors

    def idx2word(self, idx=None) -> str or list:
        if idx:
            return self.model.index2word[idx]
        return self.model.index2word

    def word2vec(self, word=None) -> dict:
        if word and word in self.model.vocab:
            idx = self.model.vocab[word].index
            return {word: self.get_vectors()[idx]}
        return {word: vec for word, vec in zip(self.idx2word(), self.get_vectors())}
