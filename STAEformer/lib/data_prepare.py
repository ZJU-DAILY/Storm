import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange

# ! X shape: (B, T, N, C)


def get_dataloaders_from_index_data(
    index, train_size, val_size, data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)


    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    # if dom:
    #     features.append(3)
    data = data[..., features]

    # 2.切割数据集
    len_data = data.shape[0]
    if index != -1:
        if index == 1:
            data = data[:int(len_data * 0.3),...]
        else:
            data = data[int(len_data * (0.3 + (index - 2) * 0.175)):int(len_data * (0.3 + (index - 1) * 0.175)), ...]

    # index = np.load(os.path.join(data_dir, "index.npz"))
    # train_index = index["train"]  # (num_samples, 3)
    # val_index = index["val"]
    # test_index = index["test"]

    total_samples = data.shape[0]
    # 计算每个集合的大小
    train_samples = int(train_size * total_samples)
    val_samples = int(val_size * total_samples)
    test_samples = total_samples - train_samples - val_samples


    # 计算每个集合的起始索引
    train_start = 0
    val_start = train_start + train_samples
    test_start = val_start + val_samples

    # 生成索引数组的第一个维度
    train_index = np.arange(train_start, train_start + train_samples).reshape(-1, 1)
    val_index = np.arange(val_start, val_start + val_samples).reshape(-1, 1)
    test_index = np.arange(test_start, len(data)-24).reshape(-1, 1)

    # 在第一个维度的基础上构建后续维度
    train_index = np.hstack([train_index, train_index + 12, train_index + 24])
    val_index = np.hstack([val_index, val_index + 12, val_index + 24])
    test_index = np.hstack([test_index, test_index + 12, test_index + 24])


    # import pdb
    # pdb.set_trace()

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_train[0])

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    # print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    # print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    # print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
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

    # import pdb
    # pdb.set_trace()
    return trainset_loader, valset_loader, testset_loader, scaler
