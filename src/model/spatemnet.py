import warnings
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, window_size, dilation_factor=1):
        super().__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [1, 3, 5, 7]

        assert (
            cout % len(self.kernel_set) == 0
        ), f"Out channels: {cout}, not divisible by length of kernel_set!"
        cout = int(cout / len(self.kernel_set))

        if self.kernel_set[-1] * dilation_factor > (2 / 3) * window_size:
            warnings.warn(
                f"Largest kernel for dilation {dilation_factor} larger than 2/3*window!"
            )

        for kern in self.kernel_set:
            # Add conv layers with dynamic padding to keep output length to input_size
            out_dim_no_pad = (window_size - dilation_factor * (kern - 1) - 1) + 1
            padding = (window_size - out_dim_no_pad) / 2

            assert (
                padding % 1 == 0
            ), f"Padding, {padding}, not integer for [kernel: {kern}, dilation: {dilation_factor}]"

            self.tconv.append(
                nn.Conv1d(
                    cin,
                    cout,
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


class temporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, window_size, dilation_factor):
        super().__init__()

        self.window_size = window_size

        self.signal = dilated_inception(
            in_channels, out_channels, window_size, dilation_factor
        )
        self.weight = dilated_inception(
            in_channels, out_channels, window_size, dilation_factor
        )
        """Possibility to add ATTENTION, by normalizing weight layer with softmax"""

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Receive data in format suitable for GCNN:
        list of [dataset_features*batch,some_dimension]
        """
        make_list = False

        if type(x) == list:
            # stack along window_dimension, as 1d conv moves "horizontally"
            x = torch.stack(x, dim=2)
            make_list = True

        # Now dim:  (N,channels,window_size), ready for 1dconv
        x = self.relu(self.signal(x)) * self.sigmoid(self.weight(x))

        if make_list:
            # Create list of x again, so gcn convs accept format:
            x_out = []
            for i in range(self.window_size):
                x_out.append(x[:, :, i])
        else:
            x_out = x

        return x_out


class spatialBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        intermediate_channels,
        window_size,
        num_hops,
        normalize=False,
    ):
        super().__init__()

        self.cin = in_channels
        self.cout = out_channels
        self.cmid = intermediate_channels
        self.window = window_size  # Create graph conv per time step
        self.hops = num_hops  # How deep graph conv is

        self.gcnConvs = nn.ModuleList()

        for i in range(self.hops):
            # Determine dimensions layers
            if i == 0 and i == self.hops - 1:
                _in = self.cin
                _out = self.cout
            elif i == 0:
                _in = self.cin
                _out = self.cmid
            elif i == self.hops - 1:
                _in = self.cmid
                _out = self.cout
            else:
                _in = self.cmid
                _out = self.cmid

            _gcns = nn.ModuleList()
            for i in range(self.window):

                _gcns.append(
                    GCNConv(in_channels=_in, out_channels=_out, normalize=normalize)
                )

            self.gcnConvs.append(_gcns)
            self.activation = nn.ELU()

    def forward(self, x_list, A_list, E_list):
        x_list_out = []
        for i in range(self.window):  # For each timestep

            x = x_list[i]
            for j in range(self.hops):  # For each hop
                x = self.gcnConvs[j][i](
                    x=x, edge_index=A_list[i], edge_weight=E_list[i]
                )  # NOTE : Reverse indexing j,i
                x = self.activation(x)

            x_list_out.append(x)
        return x_list_out


class spaTemBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        window_size,
        num_hops,
        dilation_factor,
        temporal_first=True,
    ):
        super().__init__()
        self.window = window_size
        self.temporal_first = temporal_first

        if temporal_first:
            temporal_in = in_channels
            temporal_out = out_channels
            spatial_in = out_channels
            spatial_out = out_channels
        else:
            spatial_in = in_channels
            spatial_out = out_channels
            temporal_in = out_channels
            temporal_out = out_channels

        self.spatialBlock = spatialBlock(
            in_channels=spatial_in,
            out_channels=spatial_out,
            intermediate_channels=spatial_out,
            window_size=window_size,
            num_hops=num_hops,
            normalize=False,
        )

        self.temporalBlock = temporalBlock(
            in_channels=temporal_in,
            out_channels=temporal_out,
            window_size=window_size,
            dilation_factor=dilation_factor,
        )

        self.residualConn = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x_list, A_list, E_list):

        if self.temporal_first:
            _x_list = self.temporalBlock(x_list)
            _x_list = self.spatialBlock(_x_list, A_list, E_list)
        else:
            _x_list = self.spatialBlock(x_list, A_list, E_list)
            _x_list = self.temporalBlock(_x_list)

        # residual connections
        for i in range(self.window):
            _x_list[i] = _x_list[i] + x_list[i]

        return _x_list


class SpaTemNet(nn.Module):
    def __init__(
        self,
        window_size,
        horizon,
        num_series,
        intermediate_dim=32,
        num_layers=5,
        num_hops=2,
        dilation_factor=2,
        temporal_first=True,
    ):

        super().__init__()

        self.window = window_size
        self.horizon = horizon
        self.num_layers = num_layers
        self.num_series = num_series  # Number og timeseries in data

        self.x_latent_conv = torch.nn.Conv1d(
            in_channels=1, out_channels=intermediate_dim, kernel_size=1
        )

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                spaTemBlock(
                    in_channels=intermediate_dim,
                    out_channels=intermediate_dim,
                    window_size=self.window,
                    num_hops=num_hops,
                    dilation_factor=dilation_factor,
                    temporal_first=temporal_first,
                )
            )

        # Output linear layer
        self.l_out = torch.nn.Linear(
            in_features=intermediate_dim * self.window, out_features=self.horizon
        )

    def forward(self, batch):

        x_list = []
        A_list = []
        E_list = []

        for i in range(self.window):
            x = batch.__dict__["_store"][f"x_t{i}"]

            # Project into latent dimension
            # Conv1d requires data of dim3, but gcnconv of dim 2, so first un- then sqeeeze again
            x = self.x_latent_conv(x.unsqueeze(2)).squeeze(2)

            x_list.append(x)

            A_list.append(batch.__dict__["_store"][f"edge_index_t{i}"])
            E_list.append(batch.__dict__["_store"][f"edge_attr_t{i}"])

        for i in range(self.num_layers):
            x_list = self.blocks[i](x_list, A_list, E_list)

        _xdim = x_list[0].shape[0]

        # Stack for linear output layer
        xout = torch.stack(x_list, dim=2).view(_xdim, -1)
        # Data now in shape (data_features*batch,window*block_out_dim)

        xout = self.l_out(xout)  # shape now (data_features*batch,horizon)

        # y data is in dimension (horizon*batch,features) so we need to reshape

        # Have to split data to be able to look like y data
        return torch.hstack(torch.split(xout, (self.num_series))).T
