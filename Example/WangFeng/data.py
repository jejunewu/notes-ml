import os
import jieba

PROJECT_ROOT_PATH = os.path.join(os.path.abspath(__file__), '..', '..', '..')


def read_WangFeng():
    with open(os.path.join(PROJECT_ROOT_PATH, 'Datasets', 'wangfeng.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines


seg_list = jieba.cut("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 默认是精确模式
print(", ".join(seg_list))
