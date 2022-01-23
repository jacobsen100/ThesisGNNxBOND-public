from typing import Optional
from torch_geometric.typing import OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.inits import zeros
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


from scipy.sparse.linalg import eigs, eigsh

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


class LaplacianLambdaMax(BaseTransform):
    r"""Computes the highest eigenvalue of the graph Laplacian given by
    :meth:`torch_geometric.utils.get_laplacian`.

    Args:
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of the largest eigenvalue. (default: :obj:`False`)
    """

    def __init__(self, normalization=None, is_undirected=False):
        assert normalization in [None, "sym", "rw"], "Invalid normalization"
        self.normalization = normalization
        self.is_undirected = is_undirected

    def __call__(self, edge_index, edge_weight, num_nodes):

        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, self.normalization, num_nodes=num_nodes
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        eig_fn = eigs
        if self.is_undirected and self.normalization != "rw":
            eig_fn = eigsh

        lambda_max = eig_fn(L, k=1, which="LM", return_eigenvectors=False)

        return float(lambda_max.real)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(normalization={self.normalization})"


class ChebConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        num_series=0,
        normalization: Optional[str] = "sym",
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        assert K > 0
        assert normalization in [None, "sym", "rw"], "Invalid normalization"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.num_series = num_series
        self.lins = torch.nn.ModuleList(
            [
                Linear(
                    in_channels, out_channels, bias=False, weight_initializer="glorot"
                )
                for _ in range(K)
            ]
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def __norm__(
        self,
        edge_index,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes
        )

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float("inf"), 0)

        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=-1.0, num_nodes=num_nodes
        )
        assert edge_weight is not None

        return edge_index, edge_weight

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch_size=0,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ):
        """"""
        if self.normalization != "sym" and lambda_max is None:
            raise ValueError(
                "You need to pass `lambda_max` to `forward() in`"
                "case the normalization is non-symmetric."
            )

        num_nodes = x.size(self.node_dim)

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)

            # Utilize that adjacency matrix will always be = 1 everywhere, thus largest eigenvalue
            # is n/(n-1) - where n is the number of series
            # This is also the case for batch matrices, as eigenvalues are the same for all "blocks"

            # Need to infer number of series:
            # num_series = edge_index.shape[1] / x.shape[0] FORKERT

            # l_max = num_series/(num_series-1)

            # llm = LaplacianLambdaMax("sym")
            # l_max = llm(
            #    edge_index.detach(), edge_weight.detach(), self.num_series * batch_size
            # )

            # lambda_max = torch.tensor(l_max, dtype=x.dtype, device=x.device)

        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype, device=x.device)
        assert lambda_max is not None

        # Need to get indexes and weights to multiply with norm before getting norm:
        mask = edge_index[0] != edge_index[1]
        mask = mask.detach()
        diag_mask = ~mask

        full_adjacency = torch.ones_like(edge_weight, device=x.device)
        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            full_adjacency,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        # Norm removed and added self weights 2 times (positive + negative), so now have structure:
        # [connections for i!=j, diagonal(positive), diagonal(negative)]

        # Attention on norm!!
        N = int(torch.sum(mask))
        n = int(torch.sum(diag_mask))

        n1 = (
            norm[:N] * edge_weight[mask]
        )  # torch.masked_select(edge_weight[:,0],mask).view(-1,1)
        n2 = (
            norm[N : N + n] * edge_weight[diag_mask]
        )  # torch.masked_select(edge_weight[:,0],diag_mask).view(-1,1)
        n3 = (
            norm[N + n : N + 2 * n] * edge_weight[diag_mask]
        )  # torch.masked_select(edge_weight[:,0],diag_mask).view(-1,1)

        norm = torch.cat([n1, n2, n3], dim=0)

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2.0 * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, K={len(self.lins)}, "
            f"normalization={self.normalization})"
        )
