import os
import time
from functools import partial

from skopt.space import Categorical
from skopt import gp_minimize

from .base import HyperparamOptimizer


class SkoptOptimizer(HyperparamOptimizer):
    
    def __init__(self, max_iters, acq_func):
        super().__init__()
        self.max_iters = max_iters
        self.acq_func = acq_func
    
    def objective(self, checkpoints_dir, i_epoch, params):
        self.n_iters += 1
        params = [params[0], 
                  params[1], 
                  params[2], 
                  params[3], 
                  params[4], 
                  params[5]]
        ch_dir = os.path.join(checkpoints_dir, 'models', '_'.join(map(str, params)))

        # getting training time and quality
        val_loss, val_accuracy, train_time = self.checkpoint_train_time_and_quality(ch_dir, i_epoch)
        # if we've already checked this point
        if params in self.params_track:
            train_time = 0
        self.total_time += train_time
                
        self.params_track.append(params)
        self.loss_track.append(val_loss)
        self.accuracy_track.append(val_accuracy)
        
        if val_loss is None and val_accuracy is None:
            return 100
        
        if val_accuracy > self.best_accuracy:
            self.best_loss = val_loss
            self.best_accuracy = val_accuracy
            self.best_params = params
            
        return 1 - val_accuracy


    def optimize(self, checkpoints_dir, params_grid, i_epoch):
        start_time = time.time()
        
        space = [
            Categorical(params_grid['hidden_size'], name='hidden_size'),
            Categorical(params_grid['num_layers'], name='num_layers'),
            Categorical(params_grid['dropout'], name='dropout'),
            Categorical(params_grid['bidirectional'], name='bidirectional'),
            Categorical(params_grid['batch_size'], name='batch_size'),
            Categorical(params_grid['lr'], name='learning_rate'),
        ]
        
        obj = partial(self.objective, checkpoints_dir, i_epoch)
        res_gp = gp_minimize(obj, space, n_calls=self.max_iters, acq_func=self.acq_func, random_state=42)

        best_ch_dir = os.path.join(checkpoints_dir, 'models', '_'.join(map(str, self.best_params)))
        self.test_loss, self.test_accuracy, test_time = self.test_results(best_ch_dir)
        self.total_time += time.time() - start_time + test_time
        assert self.best_accuracy == 1 - res_gp['fun']

