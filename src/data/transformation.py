import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from torch import tensor, device


class Transformation:
    def __init__(self, transforms_list, cfg, device=device("cpu")):
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.start_vals_train = None
        self.start_vals_valid = None
        self.start_vals_test = None
        self.start_window_train = None
        self.start_window_valid = None
        self.start_window_test = None
        self.mean_train = None
        self.mean_valid = None
        self.mean_test = None
        self.transforms_list = transforms_list
        self.diff_values = []
        self.device = device
        self.window_size = cfg.data.window_size
        self.prediction_offset = cfg.data.prediction_offset
        self.num_series = cfg.data_set.num_series
        self.node_dim = cfg.data_set.node_dim

    def fit(self, data):
        """
        Fit transformer on training data
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def standardize(self, df):
        if self.mean is None or self.std is None:
            raise Exception("Mean or std of data type None - Use fit method first!")
        else:
            df_standard = (df - self.mean) / self.std.T
            return df_standard

    def inv_standardize(self, df):
        if self.mean is None or self.std is None:
            raise Exception("Mean or std of data type None - Use fit method first!")
        else:
            df_inv_standard = df * self.std + self.mean
            return df_inv_standard

    def log_trans(self, df):
        return np.log(df)

    def inv_log_trans(self, df):
        return np.exp(df)

    def normalize(self, df):
        if self.min is None or self.max is None:
            raise Exception("Min or Max value is None - Use fit method first!")
        else:
            return (df - self.min) / (self.max - self.min)

    def inv_normalize(self, df):
        if self.min is None or self.max is None:
            raise Exception("Min or Max value is None - Use fit method first!")
        else:
            return df * (self.max - self.min) + self.min

    def de_trend(self, df):
        if self.start_vals_train is None:
            raise Exception("Start_vals is None - Use fit method first!")
        else:
            return np.diff(df, axis=0)

    def inv_de_trend(self, df, data_set):
        if self.start_vals_train is None:
            raise Exception("Start_vals is None - Use fit method first!")
        else:
            start_vals = getattr(self, "start_vals_" + data_set)
            return np.vstack((start_vals, df)).cumsum(axis=0)

    def frac_diff(self, df, data_set):

        if data_set == "train":
            print("IN TRAIN: Making series stationary")
            # weights = np.empty([self.window_size,df.shape[1]])
            df_diff = np.empty([df.shape[0] - (self.window_size - 1), df.shape[1]])
            d_start = 0.0
            for ts_ele in tqdm(range(df.shape[1]), total=df.shape[1]):

                d_temp = d_start
                while True:
                    frac_weight = self.compute_fractional_weights(
                        d=d_temp, length=self.window_size, threshold=1e-10
                    )
                    data_diffed = self.fracDiff_single(df[:, ts_ele], frac_weight)
                    adf_result = adfuller(data_diffed)
                    p_value = adf_result[4]["5%"]
                    # print(f'Trial d: {d_temp}, test value: {adf_result[0]}, p_value: {p_value}, #weights: {frac_weight.shape}')
                    if adf_result[0] < p_value:
                        # weights[:,ts_ele] = frac_weight
                        df_diff[:, ts_ele] = data_diffed
                        self.diff_values = np.append(self.diff_values, d_temp)
                        break
                    elif round(d_temp, 2) == 1.0:
                        print("No fraction")
                        break
                    else:
                        d_temp = round(d_temp + 0.05, 2)
            return df_diff

        else:
            if len(self.diff_values) == 0:
                raise Exception(
                    "No diff values from training data - something is wrong!"
                )

            df_diff = np.empty([df.shape[0] - (self.window_size - 1), df.shape[1]])

            for ts_ele in range(df.shape[1]):
                frac_weight = self.compute_fractional_weights(
                    d=self.diff_values[ts_ele],
                    length=self.window_size,
                    threshold=1e-10,
                )
                data_diffed = self.fracDiff_single(df[:, ts_ele], frac_weight)
                df_diff[:, ts_ele] = data_diffed
            return df_diff

    def transform(self, data, data_set=None):
        """
        Transforms data according to the arguments in the list.
        No ordering done internally, so make sure list is in correct order!
        :param: data np.ndarray
            data in numpy array
        """
        if self.transforms_list is None:
            raise Exception("No transforms_list defined - Use fit method first!")

        if data_set == "valid":
            self.start_vals_valid = data[0, :]
            self.start_window_valid = data[
                0 : self.window_size * 2 - 1 + self.prediction_offset, :
            ]
        elif data_set == "test":
            self.start_vals_test = data[0, :]
            self.start_window_test = data[0 : self.window_size * 2, :]

        for transform in self.transforms_list:
            method = getattr(self, transform)

            if transform == "frac_diff":
                data = method(data, data_set)
            else:
                data = method(data)

        if data_set == "valid":
            self.mean_valid = tensor(np.mean(data, axis=0)).to(self.device)
        elif data_set == "test":
            self.mean_test = tensor(np.mean(data, axis=0)).to(self.device)
        else:
            self.mean_train = tensor(np.mean(data, axis=0)).to(self.device)

        return data

    def fit_transform(self, data):

        append_again = False
        if self.transforms_list[0] == "de_trend":
            self.start_vals_train = data[0, :]
            data = self.de_trend(data)
            self.transforms_list.pop(0)

            append_again = True
            val = "de_trend"

        if self.transforms_list[0] == "frac_diff":
            self.start_window_train = data[0 : self.window_size * 2, :]
            self.diff_values = []
            data = self.frac_diff(data, data_set="train")
            self.transforms_list.pop(0)

            append_again = True
            val = "frac_diff"

        self.fit(data)
        data = self.transform(data, data_set="train")

        if append_again:
            self.transforms_list.insert(0, val)
        return data

    def inv_transform(self, data, data_set):
        """
        :param: data np.ndarray
            data in numpy array
        """
        if self.transforms_list is None:
            raise Exception("Transforms_list is None - Use fit method first!")
        else:
            for transform in reversed(self.transforms_list):
                if transform == "de_trend":
                    method = getattr(self, "inv_" + transform, data_set)
                else:
                    method = getattr(self, "inv_" + transform)
                data = method(data)

            return data

    # Fractional differencing
    def compute_fractional_weights(
        self, d: float, length: int, threshold: float = 1e-5
    ) -> np.array:
        """
        Computes the weights for the fracitonal differend features to a given threshold.
        :param d: float, differencing factor.
        :param length: int, length of weight vector.
        :param threshold: float, threshold of wieght size.
        return a numpy array of weights that is length long.
        """
        w, k, w_curr = [1.0], 1, 1

        while k < length:
            w_curr = -w[-1] * ((d - k + 1)) / k
            if abs(w_curr) <= threshold:
                break
            w.append(w_curr)
            k += 1
        if len(w) < length:  # Special case for d = 0,1
            delta_zeros = length - len(w)
            w = np.pad(w, (0, delta_zeros), constant_values=0)
        weights = np.flip(np.array(w))
        return weights

    def fracDiff_single(self, timeseries: np.array, weights: np.array) -> np.array:
        """
        Calculate fractional difference for a single time series.
        :param timeseries: np.array, single dimension timeseries to be differenced
        :param weights: np.array, array of weights
        """
        stacked_windows = np.lib.stride_tricks.sliding_window_view(
            timeseries, weights.shape
        )
        series_diffed = np.dot(stacked_windows, weights)
        return series_diffed

    def fractional_back_transform(
        self,
        data_set: str,
        data_fracdiffed: np.array,
    ) -> np.array:
        """
        Transforms
        :param data_set: str, train, valid or test.
        :param data_differenced: np.array: data to be back transformed.
        """
        if "standardize" in self.transforms_list:
            num_series_mask = [
                i for i in range(0, self.node_dim * self.num_series, self.node_dim)
            ]
            num_series_std = self.std[num_series_mask]
            num_series_mean = self.mean[num_series_mask]
            data_fracdiffed = data_fracdiffed * num_series_std + num_series_mean

        if data_set == "test":
            original_discard_wdw = self.start_window_test[:, num_series_mask]
        elif data_set == "valid":
            original_discard_wdw = self.start_window_valid[:, num_series_mask]
        else:
            original_discard_wdw = self.start_window_train[:, num_series_mask]

        ts_to_transform = np.vstack((original_discard_wdw[:-1, :], data_fracdiffed))
        for ts in range(data_fracdiffed.shape[1]):
            frac_weights = self.compute_fractional_weights(
                self.diff_values[num_series_mask][ts], self.window_size
            )
            pos_frac_weights = -frac_weights[:-1]
            weight_size = len(pos_frac_weights)
            for i in range(len(original_discard_wdw) - 1, len(ts_to_transform)):
                ts_to_transform[i, ts] = (
                    np.dot(ts_to_transform[i - weight_size : i, ts], pos_frac_weights)
                    + data_fracdiffed[i - (len(original_discard_wdw) - 1), ts]
                )
        return ts_to_transform

    def slice_means(self, index_list, data_set):

        if data_set == "train":
            self.mean_train = self.mean_train[index_list]

        if data_set == "valid":
            self.mean_valid = self.mean_valid[index_list]

        if data_set == "test":
            self.mean_test = self.mean_test[index_list]
