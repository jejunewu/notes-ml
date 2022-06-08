import random
import numpy as np
import pandas as pd
import torch

seed = 666
random.seed(seed)


def all_data2fold(fold_num, num=10000):
    fold_data = []
    # df_all = pd.read_csv('data/train_set.csv',sep='\t', encoding='utf-8')
    df_all = pd.read_csv('data/train_set_sample.csv', sep='\t', encoding='utf-8')
    df_all_shuffle = df_all.loc[np.random.permutation(df_all.index), :].tail(num)
    texts = df_all_shuffle['text'].tolist()
    labels = df_all_shuffle['label'].astype(str).tolist()

    label2id = {}
    for id, label in enumerate(labels):
        if label in label2id:
            label2id[label].append(id)
        else:
            label2id[label] = [id]

    # for text, label in zip(texts, labels):
    #     if label in label2id:
    #         label2id[label].append(text)
    #     else:
    #         label2id[label] = [text]

    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        batch_size = int(len(data) / fold_num)
        # other 表示多出来的数据，other 的数据量是小于 fold_num 的
        other = len(data) - batch_size * fold_num
        # 把每一类对应的 index，添加到每个 fold 里面去
        for i in range(fold_num):
            # 如果 i < other，那么将一个数据添加到这一轮 batch 的数据中
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            # batch_data 是该轮 batch 对应的索引
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            all_index[i].extend(batch_data)

    batch_size = int(len(labels) / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    # 由于上面在分 batch 的过程中，每个 batch 的数据量不一样，这里是把数据平均到每个 batch
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [texts[i] for i in all_index[fold]]
        labels = [labels[i] for i in all_index[fold]]

        if num > batch_size:  # 如果大于 batch_size 那么就取 batch_size 大小的数据
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:  # 如果小于 batch_size，那么就补全到 batch_size 的大小
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_labels)

        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)
        # 这里是为了打乱数据
        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)

    return fold_data


all_data2fold(10)
