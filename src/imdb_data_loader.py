import random

import torch


class IMDBDataLoader(object):
    
    def __init__(self, data, batch_size, ft_vectors, device, seed=42):
        self.rs = random.Random(seed)
        self.data = data
        self.batch_size = batch_size
        self.ft_vectors = ft_vectors
        self.device = device
        self.emb_size = len(list(ft_vectors.values())[0])
        self.n_batches = len(data) // batch_size + 1 if len(data) % batch_size != 0 else len(data) // batch_size
        
    def collate_fn(self, batch):
        lengths = torch.tensor([len(vectors) for vectors, label in batch])
        max_len = max(lengths)

        padded_vectors_batch = torch.zeros([len(batch), max_len, self.emb_size], dtype=torch.float)
        labels_batch = torch.empty(len(batch), dtype=torch.long)
        for i, (toks, label) in enumerate(batch):
            vectors = torch.zeros([len(toks), self.emb_size], dtype=torch.float)
            for n_t, t in enumerate(toks):
                if t in self.ft_vectors:
                    if len(self.ft_vectors[t]) != 1:
                        vectors[n_t] = torch.FloatTensor(self.ft_vectors[t])
                    else:
                        vectors[n_t] = torch.rand(self.emb_size)
                else:
                    vectors[n_t] = torch.rand(self.emb_size)
            padded_vectors_batch[i, :len(vectors), :] = vectors
            labels_batch[i] = label
        return (padded_vectors_batch.to(self.device), lengths.to(self.device)), labels_batch.to(self.device)
        
    def __len__(self):
        return self.n_batches
        
    def __iter__(self):
        data = self.rs.sample(self.data, len(self.data))
        for i in range(self.n_batches - 1):
            batch = data[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.collate_fn(batch)
        batch = data[(self.n_batches - 1) * self.batch_size :]
        yield self.collate_fn(batch)

