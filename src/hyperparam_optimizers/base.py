import os
import time
from itertools import product

import torch
import jsonlines as jsonl

from ..lstm_model import LSTMModel


class HyperparamOptimizer(object):

    def __init__(self):
        self.best_params = ()
        self.best_accuracy = 0
        self.best_loss = 100
        self.total_time = 0
        self.loss_track = []
        self.accuracy_track = []
        self.params_track = []
        self.n_iters = 0
        
    def next_params(self):
        raise NotImplementedError
        
    def __len__(self):
        return self.n_iters
        
    def checkpoint_train_time(self, ch_dir, i_epoch):
        train_result_path = os.path.join(ch_dir, '.'.join(['train_result', 'jsonl']))
        with jsonl.open(train_result_path) as reader:
            train_time = 0
            for obj in reader:
                train_time += float(obj['time'])
                if int(obj['epoch']) == i_epoch:
                    break
        return train_time
    
    def test_results(self, ch_dir, i_epoch=20):
        test_result_path = os.path.join(ch_dir, f'test_result.jsonl')
        if not os.path.exists(test_result_path):
            return None, None, None
        with jsonl.open(test_result_path) as reader:
            obj = reader.read()
            return obj['test_loss'], obj['test_accuracy'], obj['test_time']
    
    def load_model(self, ch_dir, params, i_epoch, emb_size):
        ch_path = os.path.join(ch_dir, '.'.join([str(i_epoch), 'pt']))
        if not os.path.exists(ch_path):
            print(f'{ch_path} does not exist, skipping.')
            return None
        model = LSTMModel(emb_size, params[0], params[1], params[2], params[3])
        model.load_state_dict(torch.load(ch_path)()) # accidently dumped state_dict method instead of its output
        return model
    
    def optimize(self, checkpoints_dir, params_grid, i_epoch):
        start_time = time.time()
        all_params = list(product(*params_grid.values()))
        full_train_time = 0
        for params in self.next_params(all_params):
            ch_dir = os.path.join(checkpoints_dir, 'models', '_'.join(map(str, params)))
            
            # getting training time
            train_time = self.checkpoint_train_time(ch_dir, i_epoch)
            if params in self.params_track:
                train_time = 0
            full_train_time += train_time
            
            # getting test results
            test_loss, test_accuracy, test_time = self.test_results(ch_dir, i_epoch)
            if test_loss is None and test_accuracy is None and test_time is None:
                continue
            
            if params in self.params_track:
                test_time = 0
            full_train_time += test_time
            
            # keeping track of results
            self.params_track.append(params)
            self.loss_track.append(test_loss)
            self.accuracy_track.append(test_accuracy)
            
            if test_accuracy > self.best_accuracy:
                self.best_loss = test_loss
                self.best_accuracy = test_accuracy
                self.best_params = params
        self.total_time = time.time() - start_time + full_train_time

