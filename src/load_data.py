import os
import random

from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer


def vectorize(line):
    tokens = line.rstrip().split(' ')
    return tokens[0], torch.tensor(list(map(float, tokens[1:])))

def load_vectors(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    n, d = map(int, lines[0].split())
    data = {}
    for line in tqdm(lines[1:]):
        t, vector = vectorize(line)
        data[t] = vector
    return data

def prepare_data(data_path):
    random.seed(42)
    tokenizer = TweetTokenizer()
    train_data, test_data = [], []
    for root, dirs, fns in os.walk(data_path):
        if os.path.dirname(root).endswith('train'):
            data = train_data
        elif os.path.dirname(root).endswith('test'):
            data = test_data

        if os.path.basename(root) == 'pos':
            y = 1
        elif os.path.basename(root) == 'neg':
            y = 0
        else:
            continue
        for fn in tqdm(fns, desc=root):
            with open(os.path.join(root, fn)) as f:
                toks = tokenizer.tokenize(f.read())
            x = toks
            data.append([x, y])
    train_data, val_data = train_test_split(train_data, random_state=42)
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    return train_data, val_data, test_data
