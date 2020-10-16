from tqdm import tqdm

from .base import HyperparamOptimizer


class GridSearchOptimizer(HyperparamOptimizer):

    def next_params(self, all_params):
        self.n_iters = len(all_params)
        for params in tqdm(all_params, desc='Grid search params'):
            yield params

