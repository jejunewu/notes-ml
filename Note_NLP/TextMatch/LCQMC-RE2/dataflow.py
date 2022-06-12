# %%
import pandas as pd
import numpy as np
import gensim
import os
import torch
from torch.utils.data import Dataset
from NNUtils.nlp import sentences_zh
from transformers import BertTokenizer

# %%

DATA_HOME = r'D://DataModel//LCQMC'


# %%
def get_word_list(sentence) -> list:
    sentence = sentences_zh.unicode2Ascii(sentence)
    sentence = sentences_zh.traditional2Simplified(sentence)
    sentence_list = sentences_zh.segment_sentence(sentence, is_pbx=False)
    return sentence_list


def load_corpus(file_name, data_size=None):
    """

    """
    file = os.path.join(DATA_HOME, file_name)
    df = pd.read_csv(file)
    if data_size:
        df = df.head(data_size)
    # df['text_a'] = df['text_a'].apply(get_word_list)
    # df['text_b'] = df['text_b'].apply(get_word_list)

    text_a = df['text_a'].apply(lambda x: ''.join(eval(x))).to_list()
    text_b = df['text_b'].apply(lambda x: ''.join(eval(x))).to_list()
    label = df['label'].values
    return text_a, text_b, label

def load_embeddings(embdding_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(embdding_path, binary=False)
    embedding_matrix = np.zeros((len(model.index2word) + 1, model.vector_size))
    #填充向量矩阵
    for idx, word in enumerate(model.index2word):
        embedding_matrix[idx + 1] = model[word]#词向量矩阵
    return embedding_matrix

class LcqmcDataset(Dataset):
    def __init__(self, file_name, max_len, data_size=None, device=None):
        """
        Args:
            file_name (str): 数据名, eg. 'train.txt'
            max_len (int): 单句最大长度。
            data_size (int): 读入数据大小，默认None, 为所有。
            device: cpu/gpu
        """
        super(LcqmcDataset, self).__init__()
        self.text_a, self.text_b, self.label = load_corpus(file_name=file_name, data_size=data_size)
        self.max_len = max_len
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-chinese',
            cache_dir=None,
            force_download=False,
        )
        self.text_a_pt = self.batch_decode(self.text_a)
        self.text_b_pt = self.batch_decode(self.text_b)
        self.label_pt = torch.tensor(self.label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.text_a_pt['input_ids'].to(self.device), self.text_a_pt['length'].to(self.device), self.text_b_pt[
            'input_ids'].to(self.device), self.text_b_pt[
                   'length'].to(self.device), self.label_pt.to(self.device)

    def batch_decode(self, batch_text: list) -> dict:
        output_dict = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch_text,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            return_tensors='pt',
            return_length=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )
        return output_dict

    def get_tokenizer(self):
        return self.tokenizer

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def ids2tokens(self, ids: list or torch.Tensor):
        return self.tokenizer.decode(ids)

    def tokens2ids(self, tokens):
        return self.batch_decode(tokens)
