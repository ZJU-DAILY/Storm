import os
import copy
import numpy as np
import torch
import torch
import torch.utils.data
import warnings
from lib.generate_pems_data import *


# 数据标准化
class StandardScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        data = (data - self.mean) / self.std
        return data

    def inverse_transform(self, data):
        data = (data * self.std) + self.mean
        return data


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        # Disrupt the original sample.
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            # ycl_padding = np.repeat(ycl[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        # Disrupt the original sample.
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
        # self.ycl = ycl

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                # ycl_i = self.ycl[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


def load_dataset(index, dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    # Load data
    if index != -1:
        dataset_dir = os.path.join(dataset_dir, f"set_{index}")
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']  # (, timestep, num_node, feature_dim)
        data['y_' + category] = cat_data['y']  # (, timestep, num_node, feature_dim)
    # Data format
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        # 注意：这里同时对 x_train、x_val、x_test 进行了归一化，对于标签 y 并没有做归一化
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category] = data['y_' + category][..., :1]
    # print(data['x_train'].shape)
    # print(data['x_train'][:,0,:,2])
    # import pdb
    # pdb.set_trace()
    print("train:", data['x_train'].shape, " val:", data['x_val'].shape, " test:", data['x_test'].shape)
    print("train:", data['y_train'].shape, " val:", data['y_val'].shape, " test:", data['y_test'].shape)
    # 用于监督训练 teacher forcing
    # data['ycl_train'] = copy.deepcopy(data['y_train'])
    # data['ycl_train'][..., 0] = scaler.transform(data['y_train'][..., 0])
    #
    # Iterator to initialize the dataset
    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def load_test_dataset(index, dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    # Load data
    if index != -1:
        dataset_dir = os.path.join(dataset_dir, f"set_{index}")
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']  # (, timestep, num_node, feature_dim)
        data['y_' + category] = cat_data['y']  # (, timestep, num_node, feature_dim)
    # Data format
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        # 注意：这里同时对 x_train、x_val、x_test 进行了归一化，对于标签 y 并没有做归一化
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category] = data['y_' + category][..., :1]
    # print(data['x_train'].shape)
    # print(data['x_train'][:,0,:,2])
    # import pdb
    # pdb.set_trace()
    print("train:", data['x_train'].shape, " val:", data['x_val'].shape, " test:", data['x_test'].shape)
    print("train:", data['y_train'].shape, " val:", data['y_val'].shape, " test:", data['y_test'].shape)
    # 用于监督训练 teacher forcing
    # data['ycl_train'] = copy.deepcopy(data['y_train'])
    # data['ycl_train'][..., 0] = scaler.transform(data['y_train'][..., 0])
    #
    # Iterator to initialize the dataset
    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data['x_train']), torch.FloatTensor(data['y_train'])
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data['x_val']), torch.FloatTensor(data['y_val'])
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(data['x_test']), torch.FloatTensor(data['y_test'])
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler

# 测试上面的方法
# dataset_dir = '../data/METR-LA/processed/'
# batch_size = 64
# data = load_dataset(dataset_dir, batch_size, valid_batch_size=batch_size, test_batch_size=batch_size)



warnings.filterwarnings('ignore')


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(index, args, normalizer='std', tod=False, dow=False, single=False):
    # 1.加载数据集
    data = load_st_dataset(args.dataset, args.input_dim)

    # 2.切割数据集
    len_data = data.shape[0]
    if index != -1:
        if index == 1:
            data = data[:int(len_data * 0.3), :, :]
        else:
            data = data[int(len_data * (0.3 + (index - 2) * 0.175)):int(len_data * (0.3 + (index - 1) * 0.175)), :, :]

    # 3.数据归一化处理
    data, scaler = normalize_dataset(data, normalizer)

    # 4.数据集划分(训练集、验证集、测试集)
    if args.test_ratio > 1:
        train_data, val_data, test_data = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        train_data, val_data, test_data = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    # 5.滑动窗口采样
    x_tra, y_tra = Add_Window_Horizon(train_data, args.window, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(val_data, args.window, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(test_data, args.window, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    # 6.生成数据迭代器dataloader，并对训练集样本进行打乱
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_tra) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler


if __name__ == '__main__':
    import argparse

    DATASET = 'PEMSD4'
    NODE_NUM = 307
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--input_dim', default=3, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--window', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()

    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(-1, args,
                                                                               normalizer='std',
                                                                               tod=False,
                                                                               dow=False,
                                                                               single=False)
