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
    def __int__(self, corpus: list, large_embdding_path: str):
        large_embdding_loader = LoadWord2VecEmbedding(large_embdding_path)

        self.vector_size = large_embdding_loader.vector_size
