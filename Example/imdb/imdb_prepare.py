import pandas as pd
import os

SPLIT_RATION = 0.8

IMDB_PATH = os.path.join('..', '..', 'Datasets', 'imdb')
df = pd.read_csv(os.path.join(IMDB_PATH, 'IMDB Dataset.csv'))
df.columns = ['text', 'label']
df = df[['label', 'text']]
label_map = {
    'positive': 0,
    'negative': 1
}
df['label'] = df['label'].map(label_map)
print(df)

n = len(df)
n_train = int(n * SPLIT_RATION)
df_train = df.iloc[:n_train, :]
df_test = df.iloc[n_train:, :]

df_train.to_csv(os.path.join(IMDB_PATH, 'train.csv'), index=None)
df_test.to_csv(os.path.join(IMDB_PATH, 'test.csv'), index=None)
print(df_train)
print(df_test)
