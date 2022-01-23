import math
import torch
import torch.nn as nn


class StupidShitModel(nn.Module):
    def __init__(self, window_size, horizon, num_series):
        super().__init__()
        self.horizon = horizon
        self.num_series = num_series
        self.in_features = window_size * num_series
        self.out_features = num_series * horizon

        self.l = nn.Linear(in_features=self.in_features, out_features=self.out_features)

    def forward(self, x, A, E):
        x = self.l(x.reshape(-1, self.in_features))
        x = x.reshape(-1, self.horizon, self.num_series)
        return x


class MV_LSTM(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        seq_len,
        horizon,
        batch_size,
        dropout=0,
        bidirectional=True,
        num_layers=2,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size  # number of features in the hidden states
        self.output_size = output_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_layers = num_layers  # number of LSTM layers (stacked)
        self.bidirectional = bidirectional  #
        self.ReLU = nn.ReLU()

        self.l_lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        # Output: (batch_size,seq_len, num_directions * hidden_size)

        self.l_linear = torch.nn.Linear(
            self.hidden_size * self.seq_len * (self.bidirectional + 1),
            self.output_size * self.horizon,
        )

    def forward(self, x):
        batch_size = x.size(0)
        # hidden_state = torch.zeros(self.num_layers * (self.bidirectional + 1), batch_size, self.hidden_size)
        # cell_state = torch.zeros(self.num_layers * (self.bidirectional + 1), batch_size, self.hidden_size)
        # self.hidden = (hidden_state, cell_state)
        # lstm_out, _ = self.l_lstm(x, self.hidden)
        lstm_out, _ = self.l_lstm(x)
        lstm_out = self.ReLU(lstm_out)
        x = lstm_out.contiguous().view(batch_size, -1)

        linear_out = self.l_linear(x)

        return linear_out.view(batch_size, self.horizon, self.output_size)


class MV_GRU(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        seq_len,
        horizon,
        batch_size,
        dropout=0,
        bidirectional=True,
        num_layers=2,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size  # number of features in the hidden states
        self.horizon = horizon
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_layers = num_layers  # number of GRU layers (stacked)
        self.bidirectional = bidirectional  #
        self.ReLU = nn.ReLU()

        self.GRU = torch.nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        # Output: (batch_size,seq_len, num_directions * hidden_size)

        self.l_linear = torch.nn.Linear(
            self.hidden_size * self.seq_len * (self.bidirectional + 1),
            self.input_size * self.horizon,
        )

    def forward(self, x):
        batch_size = x.size(0)
        GRU_out, _ = self.GRU(x)
        GRU_out = self.ReLU(GRU_out)
        x = GRU_out.contiguous().view(batch_size, -1)

        linear_out = self.l_linear(x)

        return linear_out.view(batch_size, self.horizon, self.input_size)
