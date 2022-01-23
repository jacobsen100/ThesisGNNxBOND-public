import math
import torch
import torch.nn as nn

import numpy as np


from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

from src.model.simplegnn import (
    latentCorrelation,
    DilatedBlockMultiGraph,
    SimpleBlockMultiGraph,
)


class NBEATSBlock(nn.Module):

    """
    Block that takes in x  and passes this
    through:
    - GCN conv layer
    - Conv1d layer to perform temporal convolutions

    Return x data in same shape as input
    """

    def __init__(
        self,
        window_size,
        num_series,
        prediction_horizon,
        in_channels=64,
        intermediate_dim=64,
        out_channels=64,
        conv_dilation=1,
        conv_kernel=3,
        dropout=0,
    ):
        super().__init__()

        self.window_size = window_size
        self.num_series = num_series
        self.in_channels = in_channels
        self.inter_dim = intermediate_dim
        self.out_channels = out_channels
        self.horizon = prediction_horizon

        # ST = Spatial Temporal
        self.STBlock = SimpleBlockMultiGraph(
            window_size,
            in_channels=in_channels,
            intermediate_dim=intermediate_dim,
            out_channels=out_channels,
            conv_kernel=conv_kernel,
            conv_dilation=conv_dilation,
            dropout=dropout,
        )

        # we do the same projection for each channel, so the dimension of in and out is w * n - otherwise it would be way too large
        self.l_x = nn.Linear(
            in_features=num_series * window_size, out_features=num_series * window_size
        )

        # before this: x.view(batch_size,-1)
        # self.l_y = torch.nn.Linear(
        #    in_features=self.inter_dim * self.window_size, out_features=self.horizon
        # )

        self.l_y = nn.Linear(
            in_features=self.window_size * self.num_series * self.inter_dim,
            out_features=self.num_series * self.horizon,
        )

    def forward(self, x_list, A, E, batch_size):

        x_list = self.STBlock(x_list, A, E)
        x_linear = (
            torch.cat(x_list, dim=1)
            .view(batch_size, -1, self.inter_dim)
            .permute(0, 2, 1)
            .reshape(batch_size * self.inter_dim, -1)
        )

        x = self.l_x(x_linear)
        x_list = list(
            torch.split(
                x.view(batch_size, self.inter_dim, -1)
                .permute(0, 2, 1)
                .reshape(self.num_series * batch_size, -1),
                self.inter_dim,
                dim=1,
            )
        )

        # Same transformation as in NetMultiGraph output
        _y = torch.stack(x_list, dim=2).view(batch_size * self.num_series, -1)
        """_y = self.l_y(_y)
        _y = torch.hstack(torch.split(_y, (self.num_series))).T.view(
            batch_size, self.horizon, self.num_series
        )"""

        _y = self.l_y(_y.view(batch_size, -1)).view(
            batch_size, self.horizon, self.num_series
        )

        return x_list, _y


class NBEATSNet(nn.Module):
    def __init__(
        self,
        window_size,
        num_series,
        prediction_horizon,
        backcast_weight=0,
        num_blocks=3,
        temporal_kernel_size=3,
        inter_dim=64,
        dropout=0,
        lcorr_gru_dim=64,
        lcorr_qk_dim=32,
        lcorr_softmax_dim=2,
    ):
        super().__init__()

        self.window = window_size
        self.num_series = num_series
        self.inter_dim = inter_dim
        self.horizon = prediction_horizon
        self.num_blocks = num_blocks
        self.beta = backcast_weight

        self.adj_generator = latentCorrelation(
            gru_dim=lcorr_gru_dim, qk_dim=lcorr_qk_dim, softmax_dim=lcorr_softmax_dim
        )

        self.x_to_inter_dim = nn.Conv1d(
            in_channels=1, out_channels=inter_dim, kernel_size=1
        )

        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            self.blocks.append(
                NBEATSBlock(
                    window_size,
                    num_series,
                    prediction_horizon,
                    in_channels=inter_dim,
                    intermediate_dim=inter_dim,
                    out_channels=inter_dim,
                    conv_kernel=temporal_kernel_size + (i * 2),  # 3,5,7,...
                    conv_dilation=1,
                    dropout=dropout,
                )
            )

        self.dropout = nn.Dropout(p=dropout)

        self.backcastLoss = nn.MSELoss()

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x, get_y=False):

        batch_size = x.shape[0]

        x_list = [
            self.x_to_inter_dim(_x.reshape(-1, 1, 1)).squeeze()
            for _x in torch.split(x, 1, dim=1)
        ]  # split on timesteps, so now collected by batch dimension
        As = []
        N = self.num_series

        for i in range(batch_size):
            As.append(
                np.vstack((np.repeat(np.arange(N), N), np.tile(np.arange(N), N)))
                + N * i
            )

        A = torch.tensor(np.concatenate(As, axis=1)).to(self.get_device())
        E = self.adj_generator(x).reshape(-1, 1)

        y = None
        BC_x = None  # BackCast
        OG_x = torch.cat(x_list, dim=0)
        backcast_loss = 0

        if get_y:
            y_outs = []

        for i, block in enumerate(self.blocks):
            _x_list, _y = block(x_list, A, E, batch_size)

            x = torch.cat(x_list, dim=0)
            _x = torch.cat(_x_list, dim=0)

            # Redidual
            x_list = list(torch.split(x - _x, self.num_series * batch_size, dim=0))

            # Block forecast
            if y is None:
                y = _y
            else:
                y = y + _y

            # Bastcast sum:
            if BC_x is None:
                BC_x = _x
            else:
                BC_x = BC_x + _x

            # Get y basis functions
            if get_y:
                y_outs.append(_y)

        backcast_loss = self.beta * self.backcastLoss(OG_x, BC_x)

        if get_y:
            return y_outs
        else:
            return backcast_loss, y
