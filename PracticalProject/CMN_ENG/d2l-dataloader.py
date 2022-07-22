import os
import re
import jieba

PROJECT_ROOT_PATH = os.path.abspath(__file__) + os.path.sep + os.path.join('..', '..', '..')


def read_data_nmt():
    """载入“英语－法语”数据集"""
    # data_dir = d2l.download_extract('fra-eng')
    file_path = os.path.join(PROJECT_ROOT_PATH, 'data', 'cmn-eng', 'cmn.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 删除【CC-BY 2.0 ...】
    text = re.sub('	CC\-BY 2\.0.+\&.+\)', '', text)
    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(jieba.lcut(parts[1]))
    return source, target


if __name__ == '__main__':
    raw_text = read_data_nmt()
    text = preprocess_nmt(raw_text)
    a, b = tokenize_nmt(text, num_examples=100)

    print(a)
    print(b)
    # data/cmn-eng

    seg = jieba.cut(text)
    # print(' '.join(seg))
