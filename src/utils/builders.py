import torch
import os
from omegaconf import DictConfig
from src.model.heteromodels import FlexHetNet

from torch_geometric.loader import DataLoader as PyGDataloader
from torch_geometric.data.batch import Batch as PyGBatch
from src.model.baseline import StupidShitModel, MV_LSTM, MV_GRU
from src.model.simplegnn import (
    BaselineBlockNetSingleGraph,
    BaselineBlockNetMultiGraph,
    DilatedBlockNetMultiGraph,
)
from src.model.heteromodels import FlexHetNet, BigHetNet
from src.model.spatemnet import SpaTemNet
from src.model.nbeatsnet import NBEATSNet
from src.data.dataset import TimeSeriesDataset, GraphDataset


def model_builder(cfg: DictConfig):

    name = cfg.model.name

    num_series = cfg.data_set.num_series

    try:
        node_dim = cfg.data_set.node_dim
        num_supports = cfg.data_set.num_supports
    except:
        num_supports = 0
        node_dim = 1

    if cfg.data.classification and name != "baselineblocknetmultigraph":
        raise Exception("Classification not supported for the given model.")

    if name == "stupidshitmodel":
        model = StupidShitModel(cfg.data.window_size, cfg.data.horizon, num_series)

    elif name == "mv_lstm":
        model = MV_LSTM(
            input_size=num_series * node_dim + num_supports,
            output_size=num_series,
            seq_len=cfg.data.window_size,
            horizon=cfg.data.horizon,
            batch_size=cfg.training.batch_size,
            hidden_size=cfg.model.hidden_layer_size,
            dropout=cfg.data.dropout,
            num_layers=cfg.model.number_of_layers,
            bidirectional=cfg.model.bidirectional,
        )

    elif name == "mv_gru":
        model = MV_GRU(
            input_size=num_series,
            seq_len=cfg.data.window_size,
            horizon=cfg.data.horizon,
            batch_size=cfg.training.batch_size,
            hidden_size=cfg.model.hidden_layer_size,
            dropout=cfg.data.dropout,
            num_layers=cfg.model.number_of_layers,
            bidirectional=cfg.model.bidirectional,
        )

    elif name == "baselineblocknetmultigraph":
        model = BaselineBlockNetMultiGraph(
            cfg.data.window_size,
            cfg.data.horizon,
            num_series,
            inter_dim=cfg.model.hidden_dim,
            num_blocks=cfg.model.num_blocks,
            temporal_kernel_size=cfg.model.temporal_kernel_init_size,
            dropout=cfg.data.dropout,
            lcorr_gru_dim=cfg.model.lcorr_gru_dim,
            lcorr_qk_dim=cfg.model.lcorr_qk_dim,
            lcorr_softmax_dim=cfg.model.lcorr_softmax_dim,
            residual_conns=cfg.model.residual_conns,
            cheb_conv_k=cfg.model.cheb_conv_k,
            classification=cfg.data.classification,
        )

    elif name == "baselineblocknetsinglegraph":
        model = BaselineBlockNetSingleGraph(
            cfg.data.window_size,
            num_series,
            cfg.data.horizon,
            num_blocks=cfg.model.num_blocks,
            inter_dim=cfg.model.hidden_dim,
            latent_adjacency=cfg.model.latent_adjacency,
            temporal_kernel_size=cfg.model.temporal_kernel_init_size,
            dropout=cfg.data.dropout,
            lcorr_gru_dim=cfg.model.lcorr_gru_dim,
            lcorr_qk_dim=cfg.model.lcorr_qk_dim,
            lcorr_softmax_dim=cfg.model.lcorr_softmax_dim,
        )

    elif name == "nbeatsnet":
        model = NBEATSNet(
            cfg.data.window_size,
            num_series,
            cfg.data.horizon,
            cfg.model.backcast_weight,
            num_blocks=cfg.model.num_blocks,
            temporal_kernel_size=cfg.model.temporal_kernel_init_size,
            inter_dim=cfg.model.hidden_dim,
            dropout=cfg.data.dropout,
            lcorr_gru_dim=cfg.model.lcorr_gru_dim,
            lcorr_qk_dim=cfg.model.lcorr_qk_dim,
            lcorr_softmax_dim=cfg.model.lcorr_softmax_dim,
        )

    elif name == "dilatedblocknetmultigraph":
        model = DilatedBlockNetMultiGraph(
            cfg.data.window_size,
            cfg.data.horizon,
            num_series,
            inter_dim=cfg.model.hidden_dim,
            num_blocks=cfg.model.num_blocks,
            dropout=cfg.data.dropout,
            lcorr_gru_dim=cfg.model.lcorr_gru_dim,
            lcorr_qk_dim=cfg.model.lcorr_qk_dim,
            lcorr_softmax_dim=cfg.model.lcorr_softmax_dim,
            residual_conns=cfg.model.residual_conns,
            no_dilation=cfg.model.no_dilation,
        )

    elif name == "flexhetnet":
        model = FlexHetNet(
            cfg.data.window_size,
            num_series,
            node_dim,
            num_supports,
            cfg.data.horizon,
            cfg.model.num_blocks,
            cfg.model.hidden_dim,
            cfg.data.dropout,
            cfg.model.lcorr_gru_dim,
            cfg.model.lcorr_qk_dim,
            cheb_conv_k=cfg.model.cheb_conv_k,
        )

    elif name == "bighetnet":
        model = BigHetNet(
            cfg.data.window_size,
            num_series,
            node_dim,
            num_supports,
            cfg.data.horizon,
            cfg.model.num_blocks,
            cfg.model.hidden_dim,
            cfg.data.dropout,
            cfg.model.lcorr_gru_dim,
            cfg.model.lcorr_qk_dim,
            cfg.model.attention_output_dim,
            cfg.model.cheb_conv_k,
            cfg.model.batch_norm,
        )

    else:
        print("[model_builder] Model is None!")
        model = None

    return model


def optimizer_builder(cfg: DictConfig, model_parameters):

    if cfg.optimizer.name == "sgd":
        optim = torch.optim.SGD(
            model_parameters,
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.name == "adam":
        optim = torch.optim.Adam(
            model_parameters,
            lr=cfg.optimizer.lr,
            betas=(cfg.optimizer.betas_low, cfg.optimizer.betas_hi),
            weight_decay=cfg.optimizer.weight_decay,
        )

    else:
        print("[optimizer_builder] Optimizer is None!")
        optim = None

    return optim


def transforms_builder(cfg: DictConfig):
    transforms = []

    if cfg.data.transforms.de_trend == 1:
        transforms.append("de_trend")
        if cfg.data.transforms.frac_diff == 1:
            raise Exception(
                "Cannot both perform de_trending and fractional_differencing!"
            )
    if cfg.data.transforms.frac_diff == 1:
        transforms.append("frac_diff")
    if cfg.data.transforms.standardize == 1:
        transforms.append("standardize")
    if cfg.data.transforms.normalize == 1:
        transforms.append("normalize")
    if cfg.data.transforms.log_trans == 1:
        transforms.append("log_trans")

    return transforms


def dataloader_builder(
    cfg: DictConfig,
    transformer,
    OG_path,
    data_set,
    shuffle=None,  # Used for test loader where we dont want to shuffle
    batch_size=None,  # Used for test loader wheere batch size 1 makes easier for plotting
):
    """
    Takes in cfg and outputs a dataloader on the correct dataset for the given model
    data_set is either test,train,valid
    """
    window_size = cfg.data.window_size
    corr_window = window_size
    horizon = cfg.data.horizon
    shuffle = cfg.data.shuffle_loader if shuffle is None else shuffle
    batch_size = cfg.training.batch_size if batch_size is None else batch_size
    corr_threshold = cfg.data.correlation_threshold

    prediction_offset = cfg.data.prediction_offset

    try:
        node_dim = cfg.data_set.node_dim
        num_supports = cfg.data_set.num_supports
    except:
        num_supports = 0
        node_dim = 1

    if cfg.model.dataset == "timeseriesdataset":
        dataset = TimeSeriesDataset(
            os.path.join(
                OG_path, "./data/interim/", cfg.data_set.name, (data_set + ".txt")
            ),
            data_set=data_set,
            window_size=window_size,
            prediction_horizon=horizon,
            transformer=transformer,
            interval=1,
            pred_offset=prediction_offset,
            num_series=cfg.data_set.num_series,
            node_dim=node_dim,
            num_supports=num_supports,
            classification=cfg.data.classification,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    elif cfg.model.dataset == "graphdataset":
        # dataname = os.path.split(cfg.data_set.path)[-1]
        dataname = cfg.data_set.name
        experiment_name = (
            f"w:{window_size}-h:{horizon}-cw:{corr_window}-ct:{int(corr_threshold*100)}"
        )
        processed_path = os.path.join(
            OG_path, "data/processed", dataname, experiment_name, data_set
        )

        dataset = GraphDataset(
            os.path.join(
                OG_path, "./data/interim/", cfg.data_set.name, (data_set + ".txt")
            ),
            data_set=data_set,
            processed_path=processed_path,
            window_size=window_size,
            prediction_horizon=horizon,
            interval=1,
            corr_threshold=corr_threshold,
            transformer=transformer,
        )
        # Tells the loader how to index graphs in dataelement
        follow_batch = [f"x_t{i}" for i in range(cfg.data.window_size)]

        loader = PyGDataloader(
            dataset,
            batch_size=batch_size,
            follow_batch=follow_batch,
            shuffle=shuffle,
        )
    return loader, dataset.num_series
