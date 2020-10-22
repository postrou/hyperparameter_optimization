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
        self.test_accuracy = 0
        self.test_loss = 100
        self.total_time = 0
        self.train_time_track = []
        self.loss_track = []
        self.accuracy_track = []
        self.best_accuracy_track = []
        self.params_track = []
        self.n_iters = 0
        
    def next_params(self):
        raise NotImplementedError
        
    def __len__(self):
        return self.n_iters
        
    def checkpoint_train_time_and_quality(self, ch_dir, i_epoch):
        train_result_path = os.path.join(ch_dir, '.'.join(['train_result', 'jsonl']))
        if not os.path.exists(os.path.join(ch_dir, f'{i_epoch}.pt')):
            val_loss, val_accuracy = None, None
        with jsonl.open(train_result_path) as reader:
            train_time = 0
            for obj in reader:
                train_time += float(obj['time'])
                if int(obj['epoch']) == i_epoch:
                    val_loss = obj['val_loss']
                    val_accuracy = obj['val_accuracy']
                    break
        return val_loss, val_accuracy, train_time
    
    def test_results(self, ch_dir):
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
        for params in self.next_params(all_params):
            ch_dir = os.path.join(checkpoints_dir, 'models', '_'.join(map(str, params)))
            
            # getting training time and validation quality
            val_loss, val_accuracy, train_time = self.checkpoint_train_time_and_quality(ch_dir, i_epoch)
            if params in self.params_track:
                train_time = 0
            self.total_time += train_time
            self.train_time_track.append(self.total_time)

            # keeping track of results
            self.params_track.append(params)
            self.loss_track.append(val_loss)
            self.accuracy_track.append(val_accuracy)
            
            if val_loss is not None and val_accuracy is not None:
                if val_accuracy > self.best_accuracy:
                    self.best_loss = val_loss
                    self.best_accuracy = val_accuracy
                    self.best_params = params
            self.best_accuracy_track.append(self.best_accuracy)

        best_ch_dir = os.path.join(checkpoints_dir, 'models', '_'.join(map(str, self.best_params)))
        self.test_loss, self.test_accuracy, test_time = self.test_results(best_ch_dir)
        self.total_time += time.time() - start_time + test_time

