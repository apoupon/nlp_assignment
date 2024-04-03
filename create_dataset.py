# Import
import transformers
import tokenizers
import datasets
import pandas as pd
import numpy as np

# load csv files
df_train = pd.read_csv('data/traindata.csv', delimiter='\t', names=['polarity', 'aspect', 'term', 'term_pos', 'sentence'])
df_dev = pd.read_csv('data/devdata.csv', delimiter='\t', names=['polarity', 'aspect', 'term', 'term_pos', 'sentence'])
pd.set_option("display.max_colwidth", None)

# create a new data frame with 3 features: sentence, pseudo-sentence (<term> <aspect>) and label (polarity)
for row in df_train.itertuples():
    df_train.at[row.Index, 'pseudo_sentence'] = row.term + ' ' + row.aspect
    df_train = df_train[['sentence', 'pseudo_sentence', 'polarity']]
    if row.polarity == 'negative':
        df_train.at[row.Index, 'polarity'] = 0
    elif row.polarity == 'neutral':
        df_train.at[row.Index, 'polarity'] = 1
    elif row.polarity == 'positive':
        df_train.at[row.Index, 'polarity'] = 2
for row in df_dev.itertuples():
    df_dev.at[row.Index, 'pseudo_sentence'] = row.term + ' ' + row.aspect
    df_dev = df_dev[['sentence', 'pseudo_sentence', 'polarity']]
    if row.polarity == 'negative':
        df_train.at[row.Index, 'polarity'] = 0
    elif row.polarity == 'neutral':
        df_train.at[row.Index, 'polarity'] = 1
    elif row.polarity == 'positive':
        df_train.at[row.Index, 'polarity'] = 2

df_train = df_train.rename(columns={'polarity': 'label'})
df_dev = df_dev.rename(columns={'polarity': 'label'})

# convert the data frame to a Dataset
train_dataset = datasets.Dataset.from_pandas(df_train)
dev_dataset = datasets.Dataset.from_pandas(df_dev)

# combine the train and dev datasets
sft_dataset = datasets.DatasetDict({'train': train_dataset, 'validation': dev_dataset})

if __name__ == '__main__':
    print('sft_dataset')
    print(sft_dataset)
    print('label values')
    print(sft_dataset['train']['label'][:10])