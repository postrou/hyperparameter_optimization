import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):

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

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.linear = nn.Linear(2 * hidden_size * self.n_directions, 2)

    def forward(self, X_batch):
        X, lengths = X_batch
        X = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        batch_size = len(lengths)
        _, (h, c) = self.lstm(X)
        last_hidden = h if self.num_layers == 1 else h[-self.n_directions:, :, :]
        last_c = c if self.num_layers == 1 else c[-self.n_directions:, :, :]
        act = torch.cat([last_hidden.transpose(0, 1).reshape([batch_size, -1]), last_c.transpose(0, 1).reshape([batch_size, -1])], dim=1)
        result = self.linear(act)
        return result

