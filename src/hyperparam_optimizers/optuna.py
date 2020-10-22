import os
import time
from functools import partial

import optuna

from .base import HyperparamOptimizer


class OptunaOptimizer(HyperparamOptimizer):

    def __init__(self, max_iters):
        super().__init__()
        self.max_iters = max_iters
    
    def objective(self, checkpoints_dir, i_epoch, params_grid, trial):
        params = [
            trial.suggest_categorical('hidden_size', params_grid['hidden_size']),
            trial.suggest_categorical('num_layers', params_grid['num_layers']),
            trial.suggest_categorical('dropout', params_grid['dropout']),
            trial.suggest_categorical('bidirectional', params_grid['bidirectional']),
            trial.suggest_categorical('batch_size', params_grid['batch_size']),
            trial.suggest_categorical('lr', params_grid['lr'])
        ]
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

        if val_loss is not None and val_accuracy is not None:
            if val_accuracy > self.best_accuracy:
                self.best_loss = val_loss
                self.best_accuracy = val_accuracy
                self.best_params = params
            result = val_accuracy
        else: 
            result = 0
        self.best_accuracy_track.append(self.best_accuracy)
        return result       

    def optimize(self, checkpoints_dir, params_grid, i_epoch):
        start_time = time.time()

        obj = partial(self.objective, checkpoints_dir, i_epoch, params_grid)
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(obj, n_trials=self.max_iters, show_progress_bar=True)

        best_trial_id = study.best_trial.number
        self.n_iters = len(study.trials)
        self.best_loss = self.loss_track[best_trial_id]
        self.best_params = self.params_track[best_trial_id]

        best_ch_dir = os.path.join(checkpoints_dir, 'models', '_'.join(map(str, self.best_params)))
        self.test_loss, self.test_accuracy, test_time = self.test_results(best_ch_dir)
        self.total_time += time.time() - start_time + test_time

