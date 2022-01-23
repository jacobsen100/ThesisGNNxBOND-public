import math
import torch
import torch.nn as nn

import numpy as np


from torch_geometric.nn import GCNConv
from src.model.custom_chebconv import ChebConv
from torch_geometric.data import Data, Batch


################################## Adjacency matrix methods ##################################
class latentCorrelation(nn.Module):
    def __init__(self, gru_dim, qk_dim, softmax_dim=2):
        super().__init__()

        self.gru_dim = gru_dim
        self.qk_dim = qk_dim
        self.softmax_dim = softmax_dim

        self.gru = nn.GRU(input_size=1, hidden_size=gru_dim)

        self.wq = nn.Linear(in_features=gru_dim, out_features=qk_dim)
        self.wk = nn.Linear(in_features=gru_dim, out_features=qk_dim)

        self.relu = nn.ReLU()

    def forward(self, x):

        batch_size = x.shape[0]
        series_encodings = []
        x = x.permute(1, 0, 2).contiguous()  # N,L,Hin -> L,N,Hin (for GRU)

        for series in torch.split(x, 1, dim=2):  # split on feature=timeseries dimension
            _, ht = self.gru(series)  # Encode each timeseries separately
            series_encodings.append(ht)

        encodings = (
            torch.cat(series_encodings, dim=0).permute(1, 0, 2).contiguous()
        )  # N,L,H
        Q = self.wq(encodings)
        KT = self.wk(encodings).permute(0, 2, 1)  # Transpose
        W = torch.matmul(Q, KT) / math.sqrt(
            self.qk_dim
        )  # Removed ReLU to avoid signal getting killed
        W = torch.softmax(W, dim=self.softmax_dim)
        return W.view(batch_size, -1)  # Reshape to correct format for pytorch geomtric


class priorCorrelation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        corrs = []
        for i in range(batch_size):
            corrs.append(
                torch.corrcoef(x[i].T.detach().clone()).fill_diagonal_(1).view(1, -1)
            )  # Fill diagonal with 1
        S = torch.abs(torch.cat(corrs))  # Take absolute value to remove negative
        S[torch.isnan(S)] = 0  # Make sure no nans, just set to 0
        return S


########################################################################################################################################
################################## Single graph network, with or without latent adjacency ##################################
class SimpleBlockSingleGraph(nn.Module):

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
        in_channels=64,
        intermediate_dim=64,
        out_channels=64,
        conv_kernel=3,
        conv_dilation=1,
    ):
        super().__init__()

        self.window_size = window_size
        self.num_series = num_series
        self.in_channels = in_channels
        self.inter_dim = intermediate_dim
        self.out_channels = out_channels

        self.gcnConv = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.inter_dim,
            add_self_loops=False,
        )  # Do not add self loops, as we calculate these ourselves.

        # conv1d args
        self.conv_kernel = conv_kernel
        self.conv_dilation = conv_dilation

        padding = int(math.floor(conv_kernel / 2) * conv_dilation)

        # x need to reshaped and transposed for conv1d (moves horizontally)
        self.conv1d = nn.Conv1d(
            in_channels=self.inter_dim,
            out_channels=self.out_channels,
            padding=padding,
            kernel_size=self.conv_kernel,
            dilation=self.conv_dilation,
        )

        self.LReLU = nn.LeakyReLU()

    def to_conv_format(self, x):

        """
        #input is shape: (nodes*timesteps*batch , hidden_dim), and dim 0 have structure: (2 nodes)
        #n1_t1 (Graph_t1)
        #n2_t1 (Graph_t1)
        #n1_t2 (Graph_t2)
        #n2_t2 (Graph_t2)
        """

        """# First split on channel dimension:
        dim_splits = torch.split(x, 1, dim=1)

        # Now split in tuples (n1_t1,n2_t1) , (n1_t2,n2_t2)
        # Make list for tuple elements:
        time_splits = [torch.split(dim, self.num_series, dim=0) for dim in dim_splits]

        # Now each tuple needs to be stacked along time-dimension.
        # First horizontally stack everything, then cut in window_sz chunks, and then stack those vertically to keep correct order
        time_stacked = [
            torch.vstack(torch.split(torch.hstack(tsplit), self.window_size, dim=1))
            for tsplit in time_splits
        ]

        return torch.stack(time_stacked, dim=1)"""

        # En epoke gik fra 130 sekunder til 5 sekunder efter dette fix!

        a = torch.split(x.T, self.num_series, dim=1)

        return torch.cat(
            torch.split(torch.stack(a, dim=1), self.window_size, dim=1), dim=2
        ).permute(2, 0, 1)

    def to_gcn_format(self, x):
        x = torch.split(x, self.num_series, dim=0)
        return torch.cat(
            [torch.movedim(split, 2, 0).reshape(-1, self.out_channels) for split in x],
            dim=0,
        )

    def forward(self, x, A, E):

        x = self.gcnConv(x, A, E)
        x = self.to_conv_format(x)
        x = self.conv1d(x)

        return self.LReLU(self.to_gcn_format(x))


class BaselineBlockNetSingleGraph(nn.Module):
    def __init__(
        self,
        window_size,
        num_series,
        prediction_horizon,
        num_blocks,
        inter_dim=64,
        latent_adjacency=True,
        temporal_kernel_size=3,
        dropout=0,
        lcorr_gru_dim=64,
        lcorr_qk_dim=32,
        lcorr_softmax_dim=2,
    ):
        super().__init__()

        self.window = window_size
        self.series = num_series
        self.horizon = prediction_horizon

        self.dim = inter_dim

        if latent_adjacency:
            self.adj_generator = latentCorrelation(
                gru_dim=lcorr_gru_dim,
                qk_dim=lcorr_qk_dim,
                softmax_dim=lcorr_softmax_dim,
            )

        else:
            self.adj_generator = priorCorrelation()

        self.x_to_inter_dim = torch.nn.Conv1d(
            in_channels=1, out_channels=inter_dim, kernel_size=1
        )

        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            self.blocks.append(
                SimpleBlockSingleGraph(
                    self.window,
                    self.series,
                    in_channels=self.dim,
                    intermediate_dim=self.dim,
                    out_channels=self.dim,
                    conv_kernel=temporal_kernel_size + (i * 2),  # 3,5,7,...
                    conv_dilation=1,
                )
            )

        self.l_out = nn.Linear(
            in_features=self.window * self.series * self.dim,
            out_features=self.series * self.horizon,
        )

        self.dropout = nn.Dropout(p=dropout)

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x):

        """
        x input dim = Batch x window x series

        """
        batch_size = x.shape[0]
        N = self.series
        As = []

        for i in range(self.window):
            As.append(
                np.vstack((np.repeat(np.arange(N), N), np.tile(np.arange(N), N)))
                + (N * i)
            )

        A = torch.tensor(np.concatenate(As, axis=1)).to(self.get_device())
        E = self.adj_generator(x).repeat(1, self.window)
        x = x.reshape(batch_size, 1, -1)
        x = self.x_to_inter_dim(x).permute(0, 2, 1)

        # Morten : behøver ikke batch her jo - bare reshape alt x data til
        # Behøver vel heller ikke repetere E?? ??? ??
        batch = Batch.from_data_list(
            [Data(x=x[i], edge_index=A, edge_attr=E[i]) for i in range(batch_size)]
        )  # Construct batch for Pytorch Geometric

        x = batch.x
        for block in self.blocks:
            x = self.dropout(block(x, batch.edge_index, batch.edge_attr))

        return self.l_out(x.view(batch_size, -1)).view(
            batch_size, self.horizon, self.series
        )


########################################################################################################################################
################################## Multi graph network, with latent adjacency ##################################
class SimpleBlockMultiGraph(nn.Module):

    """
    Block that takes in x (data), A (edge_indexes) and E (edge_wattributes) and passes this
    through:
    - layer to fix negative values in E
    - GCN conv layer for each graph/timestep in x
    - Conv1d layer to perform temporal convolutions

    Return x data in same list and shape as input
    """

    def __init__(
        self,
        window_size,
        in_channels=1,
        intermediate_dim=1,
        out_channels=1,
        conv_kernel=3,
        conv_dilation=1,
        dropout=0,
    ):
        super().__init__()

        self.window_size = window_size
        self.in_channels = in_channels
        self.inter_dim = intermediate_dim
        self.out_channels = out_channels

        # conv1d args
        self.conv_kernel = conv_kernel
        self.conv_dilation = conv_dilation

        self.gcnConvs = nn.ModuleList()
        for i in range(window_size):
            self.gcnConvs.append(
                GCNConv(
                    in_channels=self.in_channels,
                    out_channels=self.inter_dim,
                    add_self_loops=False,
                )
            )

        padding = int(math.floor(conv_kernel / 2) * conv_dilation)

        # x need to reshaped and transposed for conv1d (moves horizontally)
        self.conv1d = nn.Conv1d(
            in_channels=self.inter_dim,
            out_channels=self.out_channels,
            padding=padding,
            kernel_size=self.conv_kernel,
            dilation=self.conv_dilation,
        )

        self.dropout = nn.Dropout(p=dropout)

        self.LReLU = nn.LeakyReLU()

    def forward(self, x_list, A, E):

        # Take in 3 lists, for data, edge_indexes and edge_attributes, respectively.
        # Return list of data again in same format at input.

        gcnouts = []

        for i in range(self.window_size):
            out = self.gcnConvs[i](x=x_list[i], edge_index=A, edge_weight=E)
            gcnouts.append(out)

        # list elements should have dimension: (dataset_features*batch_size,inter_dim)
        # Stack along new dimension:
        outs = torch.stack(
            gcnouts, dim=1
        )  # New dim: (dataset_features*batch_size,window_size,inter_dim)

        # Need to be transposed for conv1d to (N,channels,series_len)
        outs = torch.transpose(
            outs, 1, 2
        )  # New dim: (features*batch_size,inter_dim,window_size)
        outs = self.conv1d(outs)

        # Transpose back again:
        outs = torch.transpose(outs, 1, 2)
        outs = self.dropout(self.LReLU(outs))
        out_list = []
        for i in range(self.window_size):
            out_list.append(outs[:, i, :])

        return out_list


class BaselineBlockNetMultiGraph(torch.nn.Module):
    """
    Baseline graph network, using multi-graph spatio-temporal blocks

    """

    def __init__(
        self,
        window_size,
        prediction_horizon,
        num_series,
        inter_dim=64,
        num_blocks=3,
        temporal_kernel_size=3,
        dropout=0,
        lcorr_gru_dim=64,
        lcorr_qk_dim=32,
        lcorr_softmax_dim=2,
        residual_conns=False,
        cheb_conv_k=0,
        classification=False,
    ):
        super().__init__()
        self.window = window_size
        self.horizon = prediction_horizon
        self.rescon = residual_conns
        self.cheb_conv_k = cheb_conv_k
        self.classification = classification

        self.num_series = num_series

        self.x_to_inter_dim = torch.nn.Conv1d(
            in_channels=1, out_channels=inter_dim, kernel_size=1
        )

        self.adj_generator = latentCorrelation(
            gru_dim=lcorr_gru_dim, qk_dim=lcorr_qk_dim, softmax_dim=lcorr_softmax_dim
        )

        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            if cheb_conv_k == 0:
                self.blocks.append(
                    SimpleBlockMultiGraph(
                        self.window,
                        in_channels=inter_dim,
                        intermediate_dim=inter_dim,
                        out_channels=inter_dim,
                        conv_kernel=temporal_kernel_size + (i * 2),  # 3,5,7,...
                        conv_dilation=1,
                        dropout=dropout,
                    )
                )

            else:

                self.blocks.append(
                    ChebConvMultiGraph(
                        self.window,
                        in_channels=inter_dim,
                        intermediate_dim=inter_dim,
                        out_channels=inter_dim,
                        conv_kernel=temporal_kernel_size + (i * 2),  # 3,5,7,...
                        conv_dilation=1,
                        dropout=dropout,
                        k=cheb_conv_k,
                        num_series=num_series,
                    )
                )
        """
        self.l_out = torch.nn.Linear(
            in_features=inter_dim * self.window, out_features=self.horizon
        )"""

        self.l_out = nn.Linear(
            in_features=self.window * self.num_series * inter_dim,
            out_features=self.num_series * self.horizon,
        )

        self.sigmoid = nn.Sigmoid()

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x):

        """
        batch_size = x.shape[0]
        N = self.series

        A = np.vstack((np.repeat(np.arange(N), N), np.tile(np.arange(N), N))).to(self.get_device())
        E = self.adj_generator(x)

        x = x.reshape(batch_size, 1, -1)
        x = self.x_to_inter_dim(x).permute(0, 2, 1)
        """

        # Alle tidsskridt skal laves til et batch, så det får samme struktur som fra dataloaderen
        # Er måske lidt langsomt men sådan er det

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

        for block in self.blocks:
            if self.rescon:
                if self.cheb_conv_k > 0:
                    _x_list = block(x_list=x_list, A=A, E=E, batch_size=batch_size)
                else:
                    _x_list = block(x_list=x_list, A=A, E=E)
                x = torch.cat(x_list, dim=0)
                _x = torch.cat(_x_list, dim=0)
                x_list = list(torch.split(x + _x, self.num_series * batch_size, dim=0))
            else:
                x_list = block(x_list=x_list, A=A, E=E)

        _xdim = x_list[0].shape[0]

        # Stack for linear output layer
        xout = torch.stack(x_list, dim=2).view(_xdim, -1)
        # Data now in shape (data_features*batch,window*block_out_dim)

        """
        xout = self.l_out(xout)  # shape now (data_features*batch,horizon)

        # y data is in dimension (horizon*batch,features) so we need to reshape
        # Have to split data to be able to look like y data
        return torch.hstack(torch.split(xout, (self.num_series))).T.view(
            batch_size, self.horizon, self.num_series
        )"""

        if self.classification:
            return self.sigmoid(
                self.l_out(xout.view(batch_size, -1)).view(
                    batch_size, self.horizon, self.num_series
                )
            )

        else:
            return self.l_out(xout.view(batch_size, -1)).view(
                batch_size, self.horizon, self.num_series
            )


########################################################################################################################################
################################## Multi graph network, with latent adjacency + dilated inception ##################################
class DilatedInception(nn.Module):
    def __init__(self, cin, cout, window_size, dilation_factor=1):
        super().__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [3, 5, 7]

        # assert (
        #    cout % len(self.kernel_set) == 0
        # ), f"Out channels: {cout}, not divisible by length of kernel_set!"

        # _cout = int(cout / len(self.kernel_set))

        print(f"Dilation : {dilation_factor}")
        for kern in self.kernel_set:
            # Add conv layers with dynamic padding to keep output length to input_size
            out_dim_no_pad = (window_size - dilation_factor * (kern - 1) - 1) + 1
            padding = (window_size - out_dim_no_pad) / 2

            assert (
                padding % 1 == 0
            ), f"Padding, {padding}, not integer for [kernel: {kern}, dilation: {dilation_factor}]"

            if kern == self.kernel_set[0]:
                _cout = math.floor(cout / len(self.kernel_set)) + cout % len(
                    self.kernel_set
                )
            else:
                _cout = math.floor(cout / len(self.kernel_set))
            print(f"channel out for kernel {kern}: {_cout}")

            self.tconv.append(
                nn.Conv1d(
                    cin,
                    _cout,
                    kernel_size=kern,
                    padding=int(padding),
                    dilation=dilation_factor,
                )
            )

    def forward(self, x):
        x_list = []
        for i in range(len(self.kernel_set)):
            x_list.append(self.tconv[i](x))

        x = torch.cat(x_list, dim=1)
        return x


class DilatedBlockMultiGraph(nn.Module):

    """
    Block that takes in x (data), A (edge_indexes) and E (edge_wattributes) and passes this
    through:
    - layer to fix negative values in E
    - GCN conv layer for each graph/timestep in x
    - Conv1d layer to perform temporal convolutions

    Return x data in same list and shape as input
    """

    def __init__(
        self,
        window_size,
        in_channels=1,
        intermediate_dim=1,
        out_channels=1,
        conv_dilation=1,
        dropout=0,
    ):
        super().__init__()

        self.window_size = window_size
        self.in_channels = in_channels
        self.inter_dim = intermediate_dim
        self.out_channels = out_channels

        # conv1d args
        self.conv_dilation = conv_dilation

        self.gcnConvs = nn.ModuleList()
        for i in range(window_size):
            self.gcnConvs.append(
                GCNConv(
                    in_channels=self.in_channels,
                    out_channels=self.inter_dim,
                    add_self_loops=False,
                )
            )

        # x need to reshaped and transposed for conv1d (moves horizontally)
        self.temporalConv = DilatedInception(
            cin=in_channels,
            cout=out_channels,
            window_size=window_size,
            dilation_factor=conv_dilation,
        )

        self.dropout = nn.Dropout(p=dropout)

        self.LReLU = nn.LeakyReLU()

    def forward(self, x_list, A, E):

        # Take in 3 lists, for data, edge_indexes and edge_attributes, respectively.
        # Return list of data again in same format at input.

        gcnouts = []

        for i in range(self.window_size):
            out = self.gcnConvs[i](x=x_list[i], edge_index=A, edge_weight=E)
            gcnouts.append(out)

        # list elements should have dimension: (dataset_features*batch_size,inter_dim)
        # Stack along new dimension:
        outs = torch.stack(
            gcnouts, dim=1
        )  # New dim: (dataset_features*batch_size,window_size,inter_dim)
        # Need to be transposed for conv1d to (N,channels,series_len)
        outs = torch.transpose(
            outs, 1, 2
        )  # New dim: (features*batch_size,inter_dim,window_size)
        outs = self.temporalConv(outs)
        # Transpose back again:
        outs = torch.transpose(outs, 1, 2)
        outs = self.dropout(self.LReLU(outs))
        out_list = []
        for i in range(self.window_size):
            out_list.append(outs[:, i, :])

        return out_list


class DilatedBlockNetMultiGraph(torch.nn.Module):
    """
    Using Dilated temporal blocks graph network, using multi-graph spatio-temporal blocks

    """

    def __init__(
        self,
        window_size,
        prediction_horizon,
        num_series,
        inter_dim=64,
        num_blocks=3,
        dropout=0,
        lcorr_gru_dim=64,
        lcorr_qk_dim=32,
        lcorr_softmax_dim=2,
        residual_conns=True,
        no_dilation=False,
    ):
        super().__init__()
        self.window = window_size
        self.horizon = prediction_horizon

        self.num_series = num_series
        self.rescon = residual_conns

        self.x_to_inter_dim = torch.nn.Conv1d(
            in_channels=1, out_channels=inter_dim, kernel_size=1
        )

        self.adj_generator = latentCorrelation(
            gru_dim=lcorr_gru_dim, qk_dim=lcorr_qk_dim, softmax_dim=lcorr_softmax_dim
        )

        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            self.blocks.append(
                DilatedBlockMultiGraph(
                    self.window,
                    in_channels=inter_dim,
                    intermediate_dim=inter_dim,
                    out_channels=inter_dim,
                    conv_dilation=i + 1
                    if not no_dilation
                    else 1,  # Linearly increase dilation
                    dropout=dropout,
                )
            )

        """
        self.l_out = torch.nn.Linear(
            in_features=inter_dim * self.window, out_features=self.horizon
        )"""

        self.l_out = nn.Linear(
            in_features=self.window * self.num_series * inter_dim,
            out_features=self.num_series * self.horizon,
        )

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x):

        """
        batch_size = x.shape[0]
        N = self.series

        A = np.vstack((np.repeat(np.arange(N), N), np.tile(np.arange(N), N))).to(self.get_device())
        E = self.adj_generator(x)

        x = x.reshape(batch_size, 1, -1)
        x = self.x_to_inter_dim(x).permute(0, 2, 1)
        """

        # Alle tidsskridt skal laves til et batch, så det får samme struktur som fra dataloaderen
        # Er måske lidt langsomt men sådan er det

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

        for block in self.blocks:
            if self.rescon:
                _x_list = block(x_list=x_list, A=A, E=E)
                x = torch.cat(x_list, dim=0)
                _x = torch.cat(_x_list, dim=0)
                x_list = list(torch.split(x + _x, self.num_series * batch_size, dim=0))
            else:
                x_list = block(x_list=x_list, A=A, E=E)

        _xdim = x_list[0].shape[0]

        # Stack for linear output layer
        xout = torch.stack(x_list, dim=2).view(_xdim, -1)
        # Data now in shape (data_features*batch,window*block_out_dim)

        """
        xout = self.l_out(xout)  # shape now (data_features*batch,horizon)

        # y data is in dimension (horizon*batch,features) so we need to reshape
        # Have to split data to be able to look like y data
        return torch.hstack(torch.split(xout, (self.num_series))).T.view(
            batch_size, self.horizon, self.num_series
        )"""

        return self.l_out(xout.view(batch_size, -1)).view(
            batch_size, self.horizon, self.num_series
        )


########################################################################################################################################
##################################               MultiBlock with chebychev conv                       ##################################
class ChebConvMultiGraph(nn.Module):

    """
    Block that takes in x (data), A (edge_indexes) and E (edge_wattributes) and passes this
    through:
    - layer to fix negative values in E
    - GCN conv layer for each graph/timestep in x
    - Conv1d layer to perform temporal convolutions

    Return x data in same list and shape as input
    """

    def __init__(
        self,
        window_size,
        in_channels=1,
        intermediate_dim=1,
        out_channels=1,
        conv_kernel=3,
        conv_dilation=1,
        dropout=0,
        k=2,
        num_series=0,  # Sum of N + NS
    ):
        super().__init__()

        self.window_size = window_size
        self.in_channels = in_channels
        self.inter_dim = intermediate_dim
        self.out_channels = out_channels

        # conv1d args
        self.conv_kernel = conv_kernel
        self.conv_dilation = conv_dilation

        self.gcnConvs = nn.ModuleList()
        for i in range(window_size):
            self.gcnConvs.append(
                ChebConv(
                    in_channels=self.in_channels,
                    out_channels=self.inter_dim,
                    K=k,
                    num_series=num_series,
                )
            )

        padding = int(math.floor(conv_kernel / 2) * conv_dilation)

        # x need to reshaped and transposed for conv1d (moves horizontally)
        self.conv1d = nn.Conv1d(
            in_channels=self.inter_dim,
            out_channels=self.out_channels,
            padding=padding,
            kernel_size=self.conv_kernel,
            dilation=self.conv_dilation,
        )

        self.dropout = nn.Dropout(p=dropout)

        self.LReLU = nn.LeakyReLU()

    def forward(self, x_list, A, E, batch_size):

        # Take in 3 lists, for data, edge_indexes and edge_attributes, respectively.
        # Return list of data again in same format at input.

        gcnouts = []

        for i in range(self.window_size):
            out = self.gcnConvs[i](
                x=x_list[i], edge_index=A, edge_weight=E, batch_size=batch_size
            )
            gcnouts.append(out)

        # list elements should have dimension: (dataset_features*batch_size,inter_dim)
        # Stack along new dimension:
        outs = torch.stack(
            gcnouts, dim=1
        )  # New dim: (dataset_features*batch_size,window_size,inter_dim)

        # Need to be transposed for conv1d to (N,channels,series_len)
        outs = torch.transpose(
            outs, 1, 2
        )  # New dim: (features*batch_size,inter_dim,window_size)
        outs = self.conv1d(outs)

        # Transpose back again:
        outs = torch.transpose(outs, 1, 2)
        outs = self.dropout(self.LReLU(outs))
        out_list = []
        for i in range(self.window_size):
            out_list.append(outs[:, i, :])

        return out_list
