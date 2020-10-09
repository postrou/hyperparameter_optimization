import os
import random
import time
from itertools import product
import sys

import numpy as np
import torch

from src.train_test import train_model, test_model
from src.load_data import load_vectors, prepare_data


def random_search(params_grid, trials, train_data, val_data, test_data, ft_vectors, epochs, emb_size, device, seed=42):
    rs = random.Random(seed)
    results = []
    start_time = time.time()
    total_time = 0
    all_params = list(product(*params_grid.values()))
    for i in range(trials):
        params = rs.choice(all_params)
        params_dict = {
            'hidden_size': int(params[0]),
            'num_layers': int(params[1]),
            'dropout': 0 if int(params[1]) == 1 else float(params[2]),
            'bidirectional': params[3],
            'batch_size': int(params[4]),
            'lr': params[5],
        }
        print(f'Trials [{i + 1}/{trials}], {params_dict}')
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        checkpoints_dir = os.path.join('checkpoints', 'random_search')
        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        checkpoints_dir = os.path.join(checkpoints_dir, '_'.join(map(str, params)))    
        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        elif os.path.exists(checkpoints_dir):
        
        model, fin_val_loss, fin_val_acc = train_model(params_dict, train_data, val_data, ft_vectors, epochs, emb_size, device, checkpoints_dir, 100)
        if model is None and fin_val_loss is None and fin_val_acc is None:
            print('Loss is None, bad trial!', end='\n\n')
            continue
        _, test_loss, test_acc = test_model(model, params_dict['batch_size'], test_data, ft_vectors, device, checkpoints_dir)
        
        print(f'Trials [{i + 1}/{trials}], test loss: {test_loss}, test accuracy: {test_acc}', end='\n\n')
        
        results.append((params, fin_val_loss, fin_val_acc, test_loss, test_acc))
        total_time += time.time() - start_time
        
    return total_time, results


if __name__ == '__main__':
    trials, epochs, device = sys.argv[1:]
    trials = int(trials)
    epochs = int(epochs)

    params_grid = {
        'hidden_size': [64, 128, 256],
        'num_layers': np.array([1, 2]),
        'dropout': np.array([0, 0.5]),
        'bidirectional': np.array([True, False]),
        'batch_size': np.array([64, 128, 256]),
        'lr': [1e-3, 1e-2, 1e-1]
    }

    print('Loading fasttext vectors')
    ft_vectors = load_vectors('wiki-news-300d-1M.vec')
    emb_size = len(list(ft_vectors.values())[0])
    print('Loading IMDB data')
    train_data, val_data, test_data = prepare_data('data/aclImdb/')

    total_time, results = random_search(params_grid, trials, train_data, val_data, test_data, ft_vectors, epochs, emb_size, device)
    print(f'Total time: {total_time}')

