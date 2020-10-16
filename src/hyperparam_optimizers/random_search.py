import random

from tqdm import trange

from .base import HyperparamOptimizer


class RandomSearchOptimizer(HyperparamOptimizer):

    def __init__(self, n_iters):
        super().__init__()
        self.n_iters = n_iters
    
    def next_params(self, all_params):
        rs = random.Random(42)
        for _ in trange(self.n_iters, desc='Random search iters'):
            yield rs.choice(all_params)

