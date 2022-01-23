import math

import torch
import torch.nn as nn
import numpy as np


from src.model.simplegnn import SimpleBlockMultiGraph, ChebConvMultiGraph


class HeteroLatentAdjacency(nn.Module):
    def __init__(
        self, gru_dim, qk_dim, num_series, node_dim, num_supports, bipartite=False
    ):
        super().__init__()

        self.num_series = num_series
        self.node_dim = node_dim
        self.num_supports = num_supports
        self.bipartite = bipartite

        self.GRU = nn.GRU(input_size=node_dim, hidden_size=gru_dim)

        if num_supports != 0:
            self.GRU_s = nn.GRU(input_size=1, hidden_size=gru_dim)

        self.wq = nn.Linear(in_features=gru_dim, out_features=qk_dim)
        self.wk = nn.Linear(in_features=gru_dim, out_features=qk_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        xb = x[:, :, : self.num_series * self.node_dim]  # x bonds
        xs = x[:, :, self.num_series * self.node_dim :]  # x supports

        # Encodings for bond nodes
        series_encodings = []
        xb = torch.cat(torch.split(xb.permute(1, 0, 2), 1, dim=2), dim=2).contiguous()
        for series in torch.split(xb, self.node_dim, dim=2):

            _, ht = self.GRU(series)  # Encode each timeseries separately
            series_encodings.append(ht)
        encodings = torch.cat(series_encodings, dim=0).permute(1, 0, 2).contiguous()

        # Encodings for support nodes
        if self.num_supports != 0:
            s_series_encodings = []

            xs = xs.permute(1, 0, 2).contiguous()  # N,L,Hin -> L,N,Hin (for GRU)
            for series in torch.split(
                xs, 1, dim=2
            ):  # split on feature=timeseries dimension
                _, ht = self.GRU_s(series)  # Encode each timeseries separately
                s_series_encodings.append(ht)

            s_encodings = (
                torch.cat(s_series_encodings, dim=0).permute(1, 0, 2).contiguous()
            )

            encodings = torch.cat((encodings, s_encodings), dim=1)

        # Get queries,keys
        Q = self.wq(encodings)
        KT = self.wk(encodings).permute(0, 2, 1)  # Transpose

        Qhom, Qhet = (
            Q[:, : self.num_series, :],
            Q[:, self.num_series :, :],
        )  # N*d_qk,NS*d_qk
        KhomT, KhetT = (
            KT[:, :, : self.num_series],
            KT[:, :, self.num_series :],
        )  # d_qk*N, d_qk*NS

        homhom = torch.matmul(Qhom, KhomT)  # N*N
        homhet = torch.matmul(Qhom, KhetT)  # N*NS
        hethom = torch.matmul(Qhet, KhomT)  # NS*N

        # Upper rows of matrix
        if self.bipartite:
            homhom = torch.sum(Qhom * KhomT.permute(0, 2, 1), dim=2, keepdim=True)
            upper = torch.cat((homhom, homhet), dim=2)
            upper = torch.softmax(upper, dim=2).view(batch_size, -1)

        else:
            upper = torch.softmax(torch.cat((homhom, homhet), dim=2), dim=2).view(
                batch_size, -1
            )

        # Lower left block of matrix
        hethet = torch.sum(Qhet * KhetT.permute(0, 2, 1), dim=2, keepdim=True)
        lower = torch.cat((hethom, hethet), dim=2)
        lower = torch.softmax(lower, dim=2).view(batch_size, -1)

        return torch.cat((upper, lower), dim=1)


class FlexHetNet(nn.Module):
    def __init__(
        self,
        window_size,
        num_series,
        node_dim,
        num_supports,
        prediction_horizon,
        num_blocks=3,
        inter_dim=64,
        dropout=0,
        lcorr_gru_dim=64,
        lcorr_qk_dim=32,
        bipartite_graph=False,
        attention_output=False,
        attention_output_dim=256,
        cheb_conv_k=1,
        batch_norm=False,
    ):

        super().__init__()
        self.window = window_size
        self.num_series = num_series
        self.node_dim = node_dim
        self.num_supports = num_supports
        self.horizon = prediction_horizon
        self.num_blocks = num_blocks
        self.inter_dim = inter_dim
        self.bipartite = bipartite_graph
        self.attention_output = attention_output
        self.cheb_conv_k = cheb_conv_k
        self.batch_norm = batch_norm

        self.adj_generator = HeteroLatentAdjacency(
            gru_dim=lcorr_gru_dim,
            qk_dim=lcorr_qk_dim,
            num_series=num_series,
            node_dim=node_dim,
            num_supports=num_supports,
            bipartite=bipartite_graph,
        )

        self.xb_to_inter_dim = torch.nn.Conv1d(
            in_channels=node_dim, out_channels=inter_dim, kernel_size=1, bias=False
        )
        self.xs_to_inter_dim = torch.nn.Conv1d(
            in_channels=1, out_channels=inter_dim, kernel_size=1, bias=False
        )

        """ with torch.no_grad():
            self.xs_to_inter_dim.weight = nn.Parameter(
                torch.ones_like(self.xs_to_inter_dim.weight)
            )

        with torch.no_grad():
            self.xb_to_inter_dim.weight = nn.Parameter(
                torch.ones_like(self.xb_to_inter_dim.weight)
            )"""

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            if cheb_conv_k == 0:
                self.blocks.append(
                    SimpleBlockMultiGraph(
                        self.window,
                        in_channels=inter_dim,
                        intermediate_dim=inter_dim,
                        out_channels=inter_dim,
                        conv_kernel=3 + (i * 2),  # 3,5,7,...
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
                        conv_kernel=3 + (i * 2),  # 3,5,7,...
                        conv_dilation=1,
                        dropout=dropout,
                        k=cheb_conv_k,
                        num_series=self.num_series + self.num_supports,
                    )
                )

        if not attention_output:
            self.l_out = nn.Linear(
                in_features=window_size * (num_series + num_supports) * inter_dim,
                out_features=num_series * prediction_horizon,
            )
        else:
            self.l_out = nn.Linear(
                in_features=window_size * (num_series + num_supports) * inter_dim,
                out_features=attention_output_dim,
            )

        self.LReLU = nn.LeakyReLU()

        self.batchnorm = torch.nn.BatchNorm1d(attention_output_dim)

    def get_device(self):
        return next(self.parameters()).device

    def get_A(self, batch_size):
        N = self.num_series
        NS = self.num_supports
        As = []

        if self.bipartite:
            for i in range(batch_size):

                abb = np.tile(np.arange(N - 1, N + NS), N)
                for j in range(N):

                    abb[(NS + 1) * j] = j
                src = np.vstack((np.repeat(np.arange(N), NS + 1), abb))
                ass = np.tile(np.arange(N + 1), NS)
                for j in range(NS):
                    ass[(N + 1) * (j) + N] = j + N

                tgt = np.vstack((np.repeat(np.arange(N, N + NS), N + 1), ass))
                As.append(np.concatenate((src, tgt), axis=1) + (N + NS) * i)

            A = torch.tensor(np.concatenate(As, axis=1)).to(self.get_device())
        else:
            for i in range(batch_size):
                abb = np.repeat(np.arange(N), N + NS)
                abss = np.repeat(np.arange(N, N + NS), N + 1)
                upper = np.concatenate((abb, abss))

                _abb = np.tile(np.arange(N + NS), N)
                _abss = np.tile(np.arange(N + 1), NS)
                for j in range(NS):
                    _abss[(N + 1) * (j) + N] = j + N
                lower = np.concatenate((_abb, _abss))

                As.append(np.stack((upper, lower), axis=0) + (N + NS) * i)

            A = torch.tensor(np.concatenate(As, axis=1)).to(self.get_device())
        return A

    def forward(self, x):
        batch_size = x.shape[0]

        assert (self.num_series * self.node_dim + self.num_supports) == x.shape[
            2
        ], f"Something wrong with data dimensions. Dim 2: {x.shape[2]}, and have: Num_series : {self.num_series}, node_dim: {self.node_dim}, num_supports: {self.num_supports}"

        # Get Adjacency index and weights
        A = self.get_A(batch_size)
        E = self.adj_generator(x).view(-1, 1)

        # Get x to correct list format for multiBlock
        xb = x[:, :, : self.num_series * self.node_dim]
        xs = x[:, :, self.num_series * self.node_dim :]

        xb_list = [
            self.xb_to_inter_dim(_x.reshape(-1, self.node_dim, 1)).squeeze()
            for _x in torch.split(xb, 1, dim=1)
        ]
        xs_list = [
            self.xs_to_inter_dim(_x.reshape(-1, 1, 1)).squeeze()
            for _x in torch.split(xs, 1, dim=1)
        ]

        if self.num_supports == 0:
            x_list = xb_list
        else:
            x_list = []
            for b, s in zip(xb_list, xs_list):
                tmp = []
                for _b, _s in zip(
                    torch.split(b, self.num_series, dim=0),
                    torch.split(s, self.num_supports, dim=0),
                ):
                    tmp.append(torch.cat((_b, _s), dim=0))
                x_list.append(torch.cat(tmp, dim=0))

        for block in self.blocks:
            if self.cheb_conv_k > 0:
                _x_list = block(x_list=x_list, A=A, E=E, batch_size=batch_size)
            else:
                _x_list = block(x_list=x_list, A=A, E=E)
            x = torch.cat(x_list, dim=0)
            _x = torch.cat(_x_list, dim=0)
            x_list = list(
                torch.split(
                    x + _x, (self.num_series + self.num_supports) * batch_size, dim=0
                )
            )

        _xdim = x_list[0].shape[0]

        # Stack for linear output layer
        xout = torch.stack(x_list, dim=2).view(_xdim, -1)
        # Data now in shape (data_features*batch,window*block_out_dim)

        xout = xout.view(batch_size, -1)
        # Now shape (batch,window*(num_series*num_supports)*inter_dim)

        if not self.attention_output:
            return self.l_out(xout).view(batch_size, self.horizon, self.num_series)
        else:
            if self.batch_norm:
                return self.batchnorm(self.LReLU(self.l_out(xout)))
            else:
                return self.LReLU(self.l_out(xout))


class BigHetNet(nn.Module):
    def __init__(
        self,
        window_size,
        num_series,
        node_dim,
        num_supports,
        prediction_horizon,
        num_blocks=3,
        inter_dim=64,
        dropout=0,
        lcorr_gru_dim=64,
        lcorr_qk_dim=32,
        attention_output_dim=256,
        cheb_conv_k=2,
        batch_norm=True,
    ):
        super().__init__()

        self.window = window_size
        self.num_series = num_series
        self.node_dim = node_dim
        self.num_supports = num_supports
        self.horizon = prediction_horizon
        self.num_blocks = num_blocks
        self.inter_dim = inter_dim
        self.attention_dim = attention_output_dim

        # Create sub networks
        self.homNet = FlexHetNet(
            window_size=window_size,
            num_series=num_series,
            node_dim=1,
            num_supports=0,
            prediction_horizon=prediction_horizon,
            num_blocks=num_blocks,
            inter_dim=inter_dim,
            dropout=dropout,
            lcorr_gru_dim=lcorr_gru_dim,
            lcorr_qk_dim=lcorr_qk_dim,
            bipartite_graph=False,
            attention_output=True,
            attention_output_dim=attention_output_dim,
            cheb_conv_k=cheb_conv_k,
            batch_norm=batch_norm,
        )

        self.auxNet = FlexHetNet(
            window_size=window_size,
            num_series=num_series,
            node_dim=node_dim,
            num_supports=0,
            prediction_horizon=prediction_horizon,
            num_blocks=num_blocks,
            inter_dim=inter_dim,
            dropout=dropout,
            lcorr_gru_dim=lcorr_gru_dim,
            lcorr_qk_dim=lcorr_qk_dim,
            bipartite_graph=False,
            attention_output=True,
            attention_output_dim=attention_output_dim,
            cheb_conv_k=cheb_conv_k,
            batch_norm=batch_norm,
        )

        self.supNet = FlexHetNet(
            window_size=window_size,
            num_series=num_series,
            node_dim=1,
            num_supports=num_supports,
            prediction_horizon=prediction_horizon,
            num_blocks=num_blocks,
            inter_dim=inter_dim,
            dropout=dropout,
            lcorr_gru_dim=lcorr_gru_dim,
            lcorr_qk_dim=lcorr_qk_dim,
            bipartite_graph=False,
            attention_output=True,
            attention_output_dim=attention_output_dim,
            cheb_conv_k=cheb_conv_k,
            batch_norm=batch_norm,
        )

        # Define data selection masks
        N = num_series * node_dim + num_supports

        # Only bond nodes
        self.hom_mask = torch.zeros(N, dtype=torch.bool, requires_grad=False)
        for i in range(num_series):
            self.hom_mask[i * node_dim] = True

        # Deselect support
        self.aux_mask = torch.zeros(N, dtype=torch.bool, requires_grad=False)
        self.aux_mask[: num_series * node_dim] = True

        # Everything except aux
        self.sup_mask = ~self.aux_mask + self.hom_mask

        # Linear layer for attention mechanism:
        self.linearAtt = nn.Linear(
            in_features=attention_output_dim,
            out_features=attention_output_dim,
            bias=False,
        )

        # Create weight vector, initialize with kaiming uniform
        self.a_weight = nn.Parameter(torch.empty(attention_output_dim, 1))
        nn.init.kaiming_uniform_(self.a_weight, a=math.sqrt(5))  # inplace operation

        factor = 3  # number of networks

        self.l_1 = nn.Linear(
            in_features=factor * attention_output_dim,
            out_features=int(factor * (attention_output_dim) * (1 / 2)),
        )

        self.l_out = nn.Linear(
            in_features=int(factor * (attention_output_dim) * (1 / 2)),
            out_features=num_series * prediction_horizon,
        )

        self.LReLU = nn.LeakyReLU()

    def forward(self, x):
        batch_size = x.shape[0]

        model_outputs = []
        model_outputs.append(self.homNet(x[:, :, self.hom_mask]))
        model_outputs.append(self.auxNet(x[:, :, self.aux_mask]))
        model_outputs.append(self.supNet(x[:, :, self.sup_mask]))

        # Attention Mechanism
        coefs = []
        linear_reprs = []
        for output in model_outputs:
            linear_repr = self.linearAtt(output)  # W.z from Multimodality article

            linear_reprs.append(linear_repr)
            coefs.append(torch.matmul(linear_repr, self.a_weight))  # aT.W.z

        softmaxed_weights = torch.split(
            torch.softmax(torch.cat(coefs, dim=1), dim=1), 1, dim=1
        )

        to_concat = []
        for weight, rep in zip(softmaxed_weights, linear_reprs):
            # rep dim: (batch, attention_output_dim),
            # weight dim: (batch,1)
            to_concat.append(torch.mul(rep, weight))

        out = torch.cat(to_concat, dim=1).contiguous()

        # Put through last linear layers and get prediction
        out = self.LReLU(self.l_1(out))

        return softmaxed_weights, self.l_out(out).view(
            batch_size, self.horizon, self.num_series
        )
