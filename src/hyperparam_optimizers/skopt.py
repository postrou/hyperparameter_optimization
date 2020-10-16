import os
import time
from functools import partial

from skopt.space import Categorical
from skopt import gp_minimize

from .base import HyperparamOptimizer


class SkoptOptimizer(HyperparamOptimizer):
    
    def __init__(self, max_iters):
        super().__init__()
        self.max_iters = max_iters
    
    def objective(self, checkpoints_dir, i_epoch, params):
        self.n_iters += 1
        params = [params[0], 
                  params[1], 
                  params[2], 
                  params[3], 
                  params[4], 
                  params[5]]
        ch_dir = os.path.join(checkpoints_dir, 'models', '_'.join(map(str, params)))

        # getting training time
        train_time = self.checkpoint_train_time(ch_dir, i_epoch)
        # if we've already checked this point
        if params in self.params_track:
            train_time = 0
        self.total_time += train_time
                
        test_loss, test_accuracy, test_time = self.test_results(ch_dir, i_epoch)
        if test_loss is None and test_accuracy is None and test_time is None:
            return 100
        
        # if we've already checked this point
        if params in self.params_track:
            test_time = 0
            
        self.params_track.append(params)
        self.loss_track.append(test_loss)
        self.accuracy_track.append(test_accuracy)
        self.total_time += test_time
        
        if test_accuracy > self.best_accuracy:
            self.best_loss = test_loss
            self.best_accuracy = test_accuracy
            self.best_params = params
            
        return 1 - test_accuracy


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
        res_gp = gp_minimize(obj, space, n_calls=self.max_iters, random_state=42)
        self.total_time += time.time() - start_time
        assert self.best_accuracy == 1 - res_gp['fun']

