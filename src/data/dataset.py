import warnings
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import Data as PyGData, InMemoryDataset as PyGDataset
from torch_geometric.data.makedirs import makedirs

from src import data


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        data_set,
        window_size,
        prediction_horizon,
        interval=1,
        transformer=None,
        pred_offset=1,
        num_series=0,
        node_dim=1,
        num_supports=0,
        classification=False,
    ):
        """
        :param data numpy.ndarray:
            2-dimensional array, dim 0: timesteps , dim 1: different timeseries
        :param: data_set str:
            string indicating if dataset is training, validation or test
        :param window_size int:
            number of historic datapoints used in prediction
        :param horizon int:
            number of datapoints to predict
        :param interval int:
            number of datapoints the rolling window skips in between data samples
        :param normalize_method object:
            transformer module, determining the preprocessing of data
        :param predOffset int:
            number of days in the future to start predicting
        :returns:
            None
        """
        self.data = np.loadtxt(data_path)
        if len(self.data.shape) != 1:
            if self.data.shape[0] < self.data.shape[1]:
                warnings.warn(
                    f"***** Warning: Data dimension 0 is: {self.data.shape[0]}, and dimension 1 is: {self.data.shape[1]}. Are you sure the orientation is correct? (dim0: timesteps, dim1: different timeseries) *****"
                )
        else:
            self.data = np.reshape(self.data, (self.data.shape[0], 1))

        self.num_series = self.data.shape[1]
        self.window = window_size
        self.horizon = prediction_horizon
        self.interval = interval
        self.transformer = transformer
        self.offset = (
            pred_offset - 1
        )  # minus because then all other logic works as before
        self.num_series = num_series
        self.node_dim = node_dim
        self.num_supports = num_supports
        self.classification = classification

        self.target_series = [i * node_dim for i in range(num_series)]

        if self.classification:
            print("################## CLASSIFICATION ##################")
            self.tgt = np.diff(self.data.copy(), n=1, axis=0)
            self.tgt[self.tgt >= 0] = 1
            self.tgt[self.tgt < 0] = 0

        if transformer:
            # If training data, fit transformer and get all data attributes
            if data_set == "train":
                self.data = transformer.fit_transform(self.data)
            else:
                # if not training data, only apply transformations, but also collect starting values for de_trending
                self.data = transformer.transform(self.data, data_set)

            transformer.slice_means(self.target_series, data_set)

        self.ts_length = self.data.shape[0]
        self.end_idx = self.ts_end_idx()

    def ts_end_idx(self):
        """
        Indexing for extracting correct rolling windows of data
        """
        index_set = range(self.window + self.offset, self.ts_length - self.horizon + 1)
        end_idx = [
            index_set[j * self.interval]
            for j in range((len(index_set)) // self.interval)
        ]
        return end_idx

    def __len__(self):
        return len(self.end_idx)

    def __getitem__(self, idx):
        # end = self.end_idx[idx]
        # start = end - self.window
        y_start = self.end_idx[idx]
        x_end = y_start - self.offset
        x_start = x_end - self.window

        train_data = self.data[x_start:x_end]

        if self.classification:
            target_data = self.tgt[y_start : y_start + self.horizon]
        else:
            target_data = self.data[y_start : y_start + self.horizon]
        x = torch.from_numpy(train_data).type(torch.float)

        if self.node_dim > 1 or self.num_supports > 0:
            y = torch.from_numpy(target_data[:, self.target_series]).type(torch.float)
        else:
            y = torch.from_numpy(target_data).type(torch.float)

        return x, y


class WindowData(PyGData):
    def __init__(self, dictionary_of_window):
        super().__init__()
        # https://stackoverflow.com/questions/60418497/how-do-i-use-kwargs-in-python-3-class-init-function
        # self.var = value, it adds it to an internal dictionary that can be accessed with self.__dict__

        # Dictionary of everything should contain all triplets of edge_index, edge_attr and x, for each
        # timestep in window

        if dictionary_of_window is not None:
            self.__dict__["_store"].update(dictionary_of_window)

    def __inc__(self, key, value, *args, **kwargs):

        # Redefine how index values for respective graphs should be incremented.
        # Assume graphs with no self loops!!!

        if key.startswith("edge_index_t"):
            i = key.split("_")[-1]
            return self.__dict__["_store"][f"x_{i}"].size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(PyGDataset):
    def __init__(
        self,
        data_path,
        data_set,
        processed_path,
        window_size,
        prediction_horizon,
        interval=1,
        corr_threshold=0,
        transformer=None,
    ):
        """
        :param data_path str:
            path to interim file that should be processed
        :param data_set str:
            train/valid/test
        :param processed_path str:
            path to the processed folder, defined by hyperparameters from config file
        """

        self.data = np.loadtxt(data_path)

        if self.data.shape[0] < self.data.shape[1]:
            warnings.warn(
                f"***** Warning: Data dimension 0 is: {self.data.shape[0]}, and dimension 1 is: {self.data.shape[1]}. Are you sure the orientation is correct? (dim0: timesteps, dim1: different timeseries) *****"
            )

        self.num_series = self.data.shape[1]
        self.window = window_size
        self.corr_window = window_size
        self.horizon = prediction_horizon
        self.interval = interval
        self.transformer = transformer
        self.corr_threshold = corr_threshold

        self.ts_length = self.data.shape[0]
        self.t_idx = self.ts_t_idx()

        self.data_path = data_path
        self.data_set = data_set
        self.processed_path = processed_path

        if transformer:
            # If training data, fit transformer and get all data attributes
            if data_set == "train":
                self.data = transformer.fit_transform(self.data)
            else:
                # if not training data, only apply transformations, but also collect starting values for de_trending
                self.data = transformer.transform(self.data, data_set)

        # Super class will perform processing,

        # super method overwrites self.data to none....
        _data = self.data
        super().__init__(root="./", transform=None, pre_transform=None)
        self.data = _data

    def _process(self):
        """Necessary to overwrite superclass method"""
        p = Path(self.processed_path)
        try:
            p.mkdir(parents=True)

        except FileExistsError:
            print(
                f"Processed folder already exist : {self.processed_path} , no processing needed!"
            )
            return

        print(f"Processing... {self.data_set} ")

        self.process()
        print("Done!")

    def ts_t_idx(self):
        """
        Indexing for extracting correct rolling windows of data
        """
        index_set = range(self.corr_window, self.ts_length - self.horizon)
        t_idx = [
            index_set[j * self.interval]
            for j in range((len(index_set)) // self.interval)
        ]
        # t_idx list of indexes timestep t sice first datapoint
        return t_idx

    def get_adj_weights(self, slice):
        # TODO: Add warning with NaNs
        Sigma = np.corrcoef(slice, rowvar=False)  # Correlation
        np.fill_diagonal(Sigma, 1)  # No numerical errors in diagonal
        Sigma = Sigma.flatten()

        Sigma[np.isnan(Sigma)] = 0  # Set timeseries with no correlation to 0
        N = slice.shape[1]  # Number of nodes
        graph = np.vstack((np.repeat(np.arange(N), N), np.tile(np.arange(N), N)))

        # mask = ~((Sigma == 1) | (np.abs(Sigma) < self.corr_threshold))
        # return graph[:, mask], Sigma[mask]
        return graph, Sigma

    def process(self):
        """
        Construct graph and pass the data to the data object
        """
        print("In process func!")

        # File is saved based on idx.
        # Idx correspond to one timestep

        for idx in tqdm(self.t_idx, total=len(self.t_idx)):

            corr_end = idx
            corr_start = corr_end - self.corr_window
            corr_slice = self.data[corr_start:corr_end]

            A, E = self.get_adj_weights(corr_slice)

            x = torch.from_numpy(self.data[idx, :]).type(torch.float).reshape(-1, 1)
            A = torch.from_numpy(A).type(torch.int64)
            E = torch.from_numpy(E).type(torch.float)
            data = PyGData(x=x, edge_index=A, edge_attr=E)

            # Do not save y data, that will be collected in self.get function
            torch.save(data, os.path.join(self.processed_path, f"data_{idx}.pt"))

    def len(self):
        return len(self.t_idx) - self.window

    def get(self, idx):
        # idx will be in range(0,len(t_idx))
        # Translate to first datapoint in prediction horizon
        _idx = idx
        idx = self.t_idx[idx]
        horizon_start = idx + self.window
        horizon_end = horizon_start + self.horizon
        """print(
            f"DATA: OG_idx : {_idx}, t_idx: {idx}, h_start : {horizon_start}, h_end : {horizon_end}"
        )"""
        y = torch.from_numpy(self.data[horizon_start:horizon_end]).type(torch.float)

        # load all files from the correct window:
        start = idx
        end = idx + self.window
        dict_of_window = {}

        for i, idx in enumerate(range(start, end)):
            _data = torch.load(
                os.path.join(self.processed_path, "data_{}.pt".format(idx))
            )

            dict_of_window[f"edge_index_t{i}"] = _data.edge_index
            dict_of_window[f"edge_attr_t{i}"] = _data.edge_attr
            dict_of_window[f"x_t{i}"] = _data.x

        dict_of_window["window_size"] = torch.tensor([self.window])
        dict_of_window["y"] = y
        data = WindowData(dict_of_window)
        return data
