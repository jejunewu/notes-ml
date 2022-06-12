import re
import unicodedata
import opencc
import jieba


def segment_sentence(sentence, mode: str = 'char', is_pbx=True) -> list:
    """
    分词
    Args:
        sentence (str):
        mode(str): 'char'按中文字分词， 'word'按中文词分词
        is_pbx(bool): 是否包含标点，True:包含， False:不包含
    """
    if not is_pbx:
        regEx_pbx = re.compile('[\\W]+')
        sentence = ''.join(regEx_pbx.split(sentence.lower()))
    if mode == 'char':
        regEx_char = re.compile(r'([\u4e00-\u9fa5])')  # [\u4e00-\u9fa5]中文范围
        sentence_list = regEx_char.split(sentence)
        sentence_list = [token for token in sentence_list if len(token) > 0]
        return sentence_list
    elif mode == 'word':
        return list(jieba.cut(sentence))


def traditional2Simplified(sentence: str) -> str:
    """
    中文繁体->简体
    """
    cc = opencc.OpenCC('t2s')
    sentence = cc.convert(sentence)
    return sentence


def unicode2Ascii(sentence: str) -> str:
    """
    编码转换 unicode -> ASCII
    Args:
        sentence(str):句子

    """
    sentence = ''.join(char for char in unicodedata.normalize('NFD', sentence)
                       if unicodedata.category(char) != 'Mn')
    return sentence
