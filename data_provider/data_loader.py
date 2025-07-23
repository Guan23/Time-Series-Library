# system
import os
import glob
import warnings
import re

# third_party
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_from_tsfile_to_dataframe

# custom
from utils.timefeatures import time_features
from utils.augmentation import run_augmentation_single
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer

warnings.filterwarnings('ignore')

__all__ = [
    "Dataset_ETT_hour",
    "Dataset_ETT_minute",
    "Dataset_Custom",
    "Dataset_M4",
    "PSMSegLoader",
    "MSLSegLoader",
    "SMAPSegLoader",
    "SMDSegLoader",
    "SWATSegLoader",
    "UEAloader",
]


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag  # 文件名标识，若指定了flag，则会re正则匹配含有flag的文件名，并只处理这些文件
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        # 获取所有.ts结尾的文件
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern = '*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        # 加载数据
        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        # 加载ts文件，用到sktime库，解析.ts时间序列文件
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                   replace_missing_vals_with='NaN')
        # 把label存成pd.Series一维序列格式
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        # 再把一维序列转成二维DataFrame，dtype从str转为int8
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        # 获取每个实例序列和其对应长度的字典（各个实例可能长度不一，后期需要统一）
        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        # 查看横向维度上的统一性，也就是每一行是否维数相同
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))
        # 如果维数都不同，则按照subsample的方法重采样，具体做法是维度除以2，隔一维一取
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        # 检查完横向维度上的一致性，再检查纵向的一致性，也就是每个实例序列的长度是否一致
        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        # 如果所有实例的序列长度不一致，获取最长序列的长度，否则随便用第一个序列的长度即可
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        # 总之，最后建立了一个二维序列，形状是(num_samples * seq_len, feat_dim)，
        # 比如共有370个序列，每个序列的长度在7~29不等，每个序列有12维
        # 则num_samples=370，seq_len=7~29不等，feat_dim=12
        # 在这一步得到的df中的values还只是原始数据，并没有做任何缺失值填充、异常值处理、统一序列长度、归一化等操作
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # 下面先做缺失值填充，这边用的方法在interpolate_missing中指定，默认linear即线性填充
        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)
        # 返回缺失值填充后的数据df，以及其对应的label，二者都是pd.DataFrame二维序列格式
        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


# TODO: 自定义数据集
class Dataset_THDY(Dataset):
    def __int__(self):
        super().__int__()

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


if __name__ == "__main__":
    import argparse


    def argparser():
        parser = argparse.ArgumentParser(description='TimesNet')

        # basic config
        parser.add_argument('--task_name', type=str, required=False, default='classification',
                            help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        # 1为训练，0为验证/测试
        parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
        parser.add_argument('--model_id', type=str, required=False, default='EthanolConcentration', help='model id')
        parser.add_argument('--model', type=str, required=False, default='TimesNet',
                            help='model name, options: [Autoformer, Transformer, TimesNet]')

        # data loader
        parser.add_argument('--data', type=str, required=False, default='UEA', help='dataset type')
        parser.add_argument('--root_path', type=str, default='G:/Time-Series-Library/dataset/JapaneseVowels',
                            help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

        # inputation task
        parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

        # anomaly detection task
        parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

        # model define
        parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
        parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
        parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
        parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size')
        parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--channel_independence', type=int, default=1,
                            help='0: channel dependence 1: channel independence for FreTS model')
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                            help='method of series decompsition, only support moving_avg or dft_decomp')
        parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
        parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
        parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
        parser.add_argument('--down_sampling_method', type=str, default=None,
                            help='down sampling method, only support avg, max, conv')
        parser.add_argument('--seg_len', type=int, default=96,
                            help='the length of segmen-wise iteration of SegRNN')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='Exp', help='exp description')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training',
                            default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        # de-stationary projector params
        parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                            help='hidden layer dimensions of projector (List)')
        parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

        # metrics (dtw)
        parser.add_argument('--use_dtw', type=bool, default=False,
                            help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

        # Augmentation
        parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
        parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
        parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
        parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
        parser.add_argument('--permutation', default=False, action="store_true",
                            help="Equal Length Permutation preset augmentation")
        parser.add_argument('--randompermutation', default=False, action="store_true",
                            help="Random Length Permutation preset augmentation")
        parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
        parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
        parser.add_argument('--windowslice', default=False, action="store_true",
                            help="Window slice preset augmentation")
        parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
        parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
        parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
        parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
        parser.add_argument('--shapedtwwarp', default=False, action="store_true",
                            help="Shape DTW warp preset augmentation")
        parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
        parser.add_argument('--discdtw', default=False, action="store_true",
                            help="Discrimitive DTW warp preset augmentation")
        parser.add_argument('--discsdtw', default=False, action="store_true",
                            help="Discrimitive shapeDTW warp preset augmentation")
        parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

        # TimeXer
        parser.add_argument('--patch_len', type=int, default=16, help='patch length')

        args = parser.parse_args()

        return args


    args = argparser()
    flag = None
    data_set = UEAloader(
        args=args,
        root_path=args.root_path,
        flag=flag,
    )

    data, label = data_set.__getitem__(1)

    print(data.shape)
    print(label.shape)
