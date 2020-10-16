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

        # getting training time
        train_time = self.checkpoint_train_time(ch_dir, i_epoch)
        # if we've already checked this point
        if params in self.params_track:
            train_time = 0
        
        test_loss, test_accuracy, test_time = self.test_results(ch_dir, i_epoch)
        if test_loss is None and test_accuracy is None and test_time is None:
            return {
                'status': STATUS_FAIL,
                'loss': None,
                'accuracy': None,
                'params': params,
                'train_time': train_time
            }
        
        # if we've already checked this point
        if params in self.params_track:
            test_time = 0
            
        self.params_track.append(params)

        return {
            'status': STATUS_OK,
            'loss': test_loss,
            'accuracy': test_accuracy,
            'params': params,
            'train_time': train_time + test_time
        }

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
        full_train_time = 0
        
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
            full_train_time += t['train_time']
            if t['status'] == 'ok':
                self.loss_track.append(t['loss'])
                self.accuracy_track.append(t['accuracy'])
                
        self.total_time = time.time() - start_time + full_train_time
        self.best_loss = trials.best_trial['result']['loss']
        self.best_accuracy = trials.best_trial['result']['accuracy']
        self.best_params = trials.best_trial['result']['params']

