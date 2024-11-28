import time

import torch
import torch.nn as nn
from random import randint
import random
import copy
import pdb
import numpy as np
import torch
from scipy.spatial import KDTree
from sklearn.manifold import MDS


# 定义数据增强函数
def get_aug_data(adj, x):
    seed = random.randint(0, 3)
    x_aug = x
    if seed == 0:
        x_aug = time_noise(adj, x)
    elif seed == 1:
        x_aug = time_shift(x, 1)
    elif seed == 2:
        x_aug = time_reversal(x)
    return x_aug


def get_aug_adj(adj, x):
    # set_env
    # 将邻接矩阵保存为txt文件
    adj_matrix_sparse = adj
    seed = random.randint(0, 3)
    if seed == 0:
        adj_matrix_sparse = aug_drop_node(adj, x, 0.1)
    elif seed == 1:
        adj_matrix_sparse = aug_drop_edges(adj, x, 0.1)
    elif seed == 2:
        adj_matrix_sparse = aug_add_edges_with_existing_weights(adj, x, 0.2)
        # adj_matrix_sparse = aug_add_edges(adj, 0.2)

    return adj_matrix_sparse

def set_env(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# def adj2dgl(weight_matrix):
#     # 将NumPy数组转换为torch张量
#     weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32)
#
#     # 使用torch的nonzero()找到非零元素的索引
#     indices = torch.nonzero(weight_matrix).t()
#     src = indices[0]  # 获取源节点索引
#     dst = indices[1]  # 获取目标节点索引
#     weights = []
#     for i in range(len(src)):
#         weights.append(weight_matrix[src[i]][dst[i]])
#     # 使用源节点索引和目标节点索引创建DGL图
#     g = dgl.graph((src, dst), num_nodes=weight_matrix.shape[0])
#
#     # 设置边的权重属性
#     g.edata['weight'] = torch.tensor(weights)
#
#     return g
#
#
# def dgl2adj(graph):
#     # 创建一个大小为 (num_nodes, num_nodes) 的零矩阵
#     # print(graph)
#     adj_matrix_with_weights = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
#
#     # 获取边的源节点、目标节点和权重
#     src_nodes, dst_nodes, _ = graph.edges(form='all')
#     weights = graph.edata['weight']
#     # 将权重填充到邻接矩阵中
#     for src, dst, weight in zip(src_nodes, dst_nodes, weights):
#         adj_matrix_with_weights[src, dst] = weight
#     # nonzero_count = np.count_nonzero(adj_matrix_with_weights)
#
#     return adj_matrix_with_weights



# 全局变量
kd_tree = None
node_positions = None
close_threshold = None
mid_threshold = None


# 定义构建KD树的函数
def build_kd_tree(adj):
    global kd_tree, node_positions, close_threshold, mid_threshold

    # 使用MDS将邻接矩阵（距离矩阵）转化为节点的二维坐标
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    symmetrize_adj = adj
    symmetrize_adj = (symmetrize_adj + symmetrize_adj.T) / 2
    node_positions = mds.fit_transform(symmetrize_adj)

    # 构建KD树
    kd_tree = KDTree(node_positions)

    # 计算距离的阈值来划分近、中、远邻居
    all_distances = []
    num_nodes = adj.shape[0]
    for i in range(num_nodes):
        distances, _ = kd_tree.query(node_positions[i], k=num_nodes)
        all_distances.extend(distances[1:])  # 排除自己与自己的距离

    # 确定阈值
    all_distances = np.array(all_distances)
    close_threshold = np.percentile(all_distances, 20)
    mid_threshold = np.percentile(all_distances, 50)


# 定义时间噪声添加函数
def time_noise(adj, x, noise_level=0.1):
    global kd_tree, node_positions, close_threshold, mid_threshold

    # 添加噪声的内部函数
    def add_noise_with_kdtree(time_series, adj, kd_tree, node_positions, close_threshold, mid_threshold,
                              noise_level=0.1):
        bs, length, num_nodes, num_features = time_series.shape
        noise = torch.randn(bs, length, num_nodes, 1) * noise_level
        noise = noise.to(x.device)
        # 使用KD树找到每个节点的所有邻居
        for i in range(num_nodes):
            distances, indices = kd_tree.query(node_positions[i], k=num_nodes)

            close_neighbors = []
            mid_neighbors = []
            far_neighbors = []

            for idx, dist in enumerate(distances[1:], 1):  # 排除自己与自己的距离
                if dist < close_threshold:
                    close_neighbors.append(indices[idx])
                elif close_threshold <= dist < mid_threshold:
                    mid_neighbors.append(indices[idx])
                else:
                    far_neighbors.append(indices[idx])
            # 从每一类中选择一个邻居并加权
            if close_neighbors:
                j = np.random.choice(close_neighbors)
                time_series[:, :, i, :1] += noise[:, :, j, :1] * 0.5
            if mid_neighbors:
                j = np.random.choice(mid_neighbors)
                time_series[:, :, i, :1] += noise[:, :, j, :1] * 0.3
            if far_neighbors:
                j = np.random.choice(far_neighbors)
                time_series[:, :, i, :1] += noise[:, :, j, :1] * 0.2

        return time_series

    # 添加噪声
    x_noisy = add_noise_with_kdtree(x.clone(), adj, kd_tree, node_positions, close_threshold, mid_threshold,
                                    noise_level)
    return x_noisy

# 定义时间平移函数
def time_shift(x, shift):
    return torch.roll(x, shifts=shift, dims=1)

def time_reversal(x):
    return torch.flip(x, dims=[1])




def calculate_time_correlation(time_series):
    """
    计算时间相关性矩阵
    :param time_series: 输入的时间序列数据 (torch.tensor)，形状为 [bs, num_features, num_nodes, length]
    :return: 时间相关性矩阵 (torch.tensor)
    """
    bs, num_features, num_nodes, length = time_series.shape
    # 计算每个节点的时间特征向量 (按时间步和特征求平均)
    time_features = time_series.mean(dim=(0, 1))  # 形状为 [num_nodes, length]

    # 减去均值以计算相关性
    time_features = time_features - time_features.mean(dim=1, keepdim=True)
    # 计算协方差矩阵
    time_corr = torch.matmul(time_features, time_features.t()) / (length - 1)
    # 计算标准差
    std = time_features.std(dim=1, keepdim=True)
    # 归一化协方差矩阵得到相关性矩阵
    time_corr = time_corr / (std @ std.t())

    return time_corr


def aug_add_edges_with_existing_weights(adj, time_series, add_percent=0.2):
    """
    添加具有现有权重的边
    :param adj: 邻接矩阵 (numpy.ndarray)
    :param time_series: 输入的时间序列数据 (torch.tensor)
    :param add_percent: 增加的边的百分比
    :return: 增强后的邻接矩阵 (numpy.ndarray)
    """
    aug_adj = adj.copy()
    node_num = aug_adj.shape[0]
    edge_indices = [(i, j) for i in range(node_num) for j in range(i)]
    current_edges = np.transpose(np.nonzero(aug_adj > 0))
    current_edges = set(tuple(edge) for edge in current_edges)
    possible_edges = list(set(edge_indices) - current_edges)
    add_num = int(len(edge_indices) * add_percent / 2)

    time_corr = calculate_time_correlation(time_series).cpu().numpy()
    existing_weights = aug_adj[aug_adj > 0]

    if add_num > 0 and len(existing_weights) > 0:
        # 根据时间相关性排序可能的边
        possible_edges.sort(key=lambda x: time_corr[x[0], x[1]], reverse=True)
        selected_edges = possible_edges[:add_num]
        for edge in selected_edges:
            weight = random.choice(existing_weights)
            aug_adj[edge[0], edge[1]] = weight
            aug_adj[edge[1], edge[0]] = weight
    return aug_adj


def aug_drop_node(adj, time_series, drop_percent=0.1):
    """
    删除节点
    :param adj: 邻接矩阵 (numpy.ndarray)
    :param time_series: 输入的时间序列数据 (torch.tensor)
    :param drop_percent: 删除的节点百分比
    :return: 增强后的邻接矩阵 (numpy.ndarray)
    """
    aug_adj = adj.copy()
    num = aug_adj.shape[0]
    drop_num = int(num * drop_percent)
    all_node_list = list(range(num))

    time_corr = calculate_time_correlation(time_series)
    node_importance = time_corr.mean(dim=1)
    node_importance_sorted = torch.argsort(node_importance).cpu().numpy()

    # 优先删除时间相关性较弱的节点
    drop_node_list = node_importance_sorted[:drop_num]

    aug_adj[drop_node_list, :] = 0
    aug_adj[:, drop_node_list] = 0

    return aug_adj


def aug_drop_edges(adj, time_series, drop_percent=0.1):
    """
    删除边
    :param adj: 邻接矩阵 (numpy.ndarray)
    :param time_series: 输入的时间序列数据 (torch.tensor)
    :param drop_percent: 删除的边百分比
    :return: 增强后的邻接矩阵 (numpy.ndarray)
    """
    aug_adj = adj.copy()
    node_num = aug_adj.shape[0]
    edge_indices = np.transpose(np.triu_indices(node_num, k=1))
    current_edges = np.transpose(np.nonzero(aug_adj > 0))
    current_edges = [tuple(edge) for edge in current_edges]
    drop_num = int(len(current_edges) * drop_percent)

    time_corr = calculate_time_correlation(time_series).cpu().numpy()

    if drop_num > 0:
        # 根据时间相关性排序当前的边
        current_edges.sort(key=lambda x: time_corr[x[0], x[1]])
        selected_edges = current_edges[:drop_num]
        for edge in selected_edges:
            aug_adj[edge[0], edge[1]] = 0
            aug_adj[edge[1], edge[0]] = 0
    return aug_adj

