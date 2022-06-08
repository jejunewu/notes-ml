import pandas as pd
import numpy as np
import re
import opencc

text_path =r'D://DataModel//cmn-eng//cmn.txt'


def read_data_cmn():
    """载入“英语－法语”数据集"""
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_cmn(text):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 删除【CC-BY 2.0 ...】
    text = re.sub('	CC-BY 2\.0.+&.+\)', '', text)
    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def split_cmn(text, ):
    """词元化“英语－法语”数据数据集"""
    en, zh = [], []
    cc = opencc.OpenCC('t2s')
    for i, line in enumerate(text.split('\n')):
        parts = line.split('\t')
        if len(parts) == 2:
            en.append(parts[0])
            zh.append(cc.convert(parts[1]))
    return en, zh

if __name__ == '__main__':
    text = read_data_cmn()
    text = preprocess_cmn(text)
    en, zh = split_cmn(text)

    df = pd.DataFrame(np.array([en,zh]).T, columns=['en', 'zh'])
    df.to_csv('cmn.csv',sep='\t')

