import os
import time
from functools import partial

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval

from .base import HyperparamOptimizer


class HyperoptOptimizer(HyperparamOptimizer):
    
    def __init__(self, max_iters):
        super().__init__()
        self.max_iters = max_iters
    
    def objective(self, checkpoints_dir, i_epoch, params):
        params = [params['hidden_size'], 
                  params['num_layers'], 
                  params['dropout'], 
                  params['bidirectional'], 
                  params['batch_size'], 
                  params['learning_rate']]
        ch_dir = os.path.join(checkpoints_dir, 'models', '_'.join(map(str, params)))

        # getting training time and quality
        val_loss, val_accuracy, train_time = self.checkpoint_train_time_and_quality(ch_dir, i_epoch)
        # if we've already checked this point
        if params in self.params_track:
            train_time = 0
        self.params_track.append(params)
        
        if val_loss is not None and val_accuracy is not None:
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy

            # here loss is (1 - accuracy), because hyperopt triggers on "loss" variable, during optimization
            result = {
                'status': STATUS_OK,
                'loss': 1 - val_accuracy,
                'cross_entropy': val_loss,
                'params': params,
                'train_time': train_time
            }

        else:
            result =  {
                'status': STATUS_FAIL,
                'loss': None,
                'cross_entropy': None,
                'params': params,
                'train_time': train_time
            }
        self.best_accuracy_track.append(self.best_accuracy)
        return result
                
    def optimize(self, checkpoints_dir, params_grid, i_epoch):
        start_time = time.time()
        space = hp.choice('net', [
            {
                'hidden_size': hp.choice('h', params_grid['hidden_size']),
                'num_layers': hp.choice('n', params_grid['num_layers']),
                'dropout': hp.choice('dp', params_grid['dropout']),
                'bidirectional': hp.choice('bd', params_grid['bidirectional']),
                'batch_size': hp.choice('bs', params_grid['batch_size']),
                'learning_rate': hp.choice('lr', params_grid['lr'])
            }
        ])
        trials = Trials()
        obj = partial(self.objective, checkpoints_dir, i_epoch)
        best = fmin(
            fn=obj,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_iters,
            trials=trials
        )
        
        self.n_iters = len(trials.results)
        for t in trials.results:
            val_accuracy = 1 - t['loss'] if t['loss'] is not None else None
            self.total_time += t['train_time']
            self.train_time_track.append(self.total_time)
            self.loss_track.append(t['cross_entropy'])
            self.accuracy_track.append(val_accuracy)
                
        self.best_loss = trials.best_trial['result']['cross_entropy']
        self.best_params = trials.best_trial['result']['params']

        best_ch_dir = os.path.join(checkpoints_dir, 'models', '_'.join(map(str, self.best_params)))
        self.test_loss, self.test_accuracy, test_time = self.test_results(best_ch_dir)
        self.total_time += time.time() - start_time + test_time

