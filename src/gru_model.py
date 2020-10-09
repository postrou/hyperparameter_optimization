import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUModel(nn.Module):
    
    def __init__(self,
                 input_size, 
                 hidden_size, 
                 num_layers, 
                 dropout,
                 bidirectional=False):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.n_directions = int(bidirectional) + 1
        
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout,
                          bidirectional=self.bidirectional,
                          batch_first=True)
        self.linear = nn.Linear(hidden_size * self.n_directions, 2)
        
    def forward(self, X_batch, lengths):
        X_batch = pack_padded_sequence(X_batch, lengths, batch_first=True, enforce_sorted=False)
        batch_size = len(lengths)
        output, h_n = self.gru(X_batch) # output: (seq_len, batch, num_directions * hidden_size)
        output, _ = pad_packed_sequence(output)
        output = output[-1, :, :].squeeze() # (1, batch, num_directions * hidden_size)
        result = self.linear(output) # mean over directions
        return result

