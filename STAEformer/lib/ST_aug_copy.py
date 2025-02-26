import time

import torch
import torch.nn as nn
from random import randint
import random
import copy
import pdb
import numpy as np
import torch


def origin_get_aug_adj(adj):
    # set_env(42)
    adj_matrix_sparse = adj
    seed = random.randint(0, 3)
    if seed == 0:
        adj_matrix_sparse = aug_drop_node(adj, 0.1)
    elif seed == 1:
        adj_matrix_sparse = aug_drop_edges(adj, 0.1)
    elif seed == 2:
        adj_matrix_sparse = aug_add_edges_with_existing_weights(adj, 0.2)
        # adj_matrix_sparse = aug_add_edges(adj, 0.2)

    return adj_matrix_sparse

def origin_get_aug_data(x):
    # set_env(42)
    seed = random.randint(0, 3)

    x_aug = x
    if seed == 0:
        x_aug = time_noise(x)
    elif seed == 1:
        x_aug = time_shift(x, 1)
    elif seed == 2:
        x_aug = time_reversal(x)
    return x_aug

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

def time_shift(x, shift):
    return torch.roll(x, shifts=shift, dims=1)

def time_noise(x, noise_level=0.1):
    noise = torch.normal(0, noise_level, size=x[..., :1].size(), device=x.device)
    x_aug = x.clone()  # 创建x的副本
    x_aug[..., :1] += noise  # 在副本上进行操作
    return x

def time_reversal(x):
    return torch.flip(x, dims=[1])






def aug_drop_node(adj, drop_percent=0.1):
    aug_adj = copy.deepcopy(adj)
    num = aug_adj.shape[0]
    drop_num = int(num * drop_percent)
    all_node_list = list(range(num))
    drop_node_list = random.sample(all_node_list, drop_num)

    aug_adj[drop_node_list, :] = 0
    aug_adj[:, drop_node_list] = 0

    return aug_adj


def aug_drop_edges(adj, drop_percent=0.1):
    aug_adj = copy.deepcopy(adj)
    node_num = aug_adj.shape[0]
    edge_indices = np.transpose(np.triu_indices(node_num, k=1))
    current_edges = np.transpose(np.where(aug_adj > 0))
    current_edges = [tuple(edge) for edge in current_edges]
    drop_num = int(len(current_edges) * drop_percent)

    if drop_num > 0:
        selected_edges = random.sample(current_edges, drop_num)
        for edge in selected_edges:
            aug_adj[edge[0], edge[1]] = 0
            aug_adj[edge[1], edge[0]] = 0
    return aug_adj


def aug_add_edges(adj, add_percent=0.2):
    aug_adj = copy.deepcopy(adj)
    node_num = aug_adj.shape[0]
    edge_indices = [(i, j) for i in range(node_num) for j in range(i)]
    current_edges = np.transpose(np.where(aug_adj > 0))
    current_edges = set(tuple(edge) for edge in current_edges)
    possible_edges = list(set(edge_indices) - current_edges)
    add_num = int(len(edge_indices) * add_percent / 2)

    if add_num > 0:
        selected_edges = random.sample(possible_edges, add_num)
        for edge in selected_edges:
            aug_adj[edge[0], edge[1]] = 1
            aug_adj[edge[1], edge[0]] = 1
    return aug_adj

def aug_add_edges_with_existing_weights(adj, add_percent=0.2):
    aug_adj = copy.deepcopy(adj)
    node_num = aug_adj.shape[0]
    edge_indices = [(i, j) for i in range(node_num) for j in range(i)]
    current_edges = np.transpose(np.where(aug_adj > 0))
    current_edges = set(tuple(edge) for edge in current_edges)
    possible_edges = list(set(edge_indices) - current_edges)
    add_num = int(len(edge_indices) * add_percent / 2)

    existing_weights = aug_adj[aug_adj > 0]
    if add_num > 0 and len(existing_weights) > 0:
        selected_edges = random.sample(possible_edges, add_num)
        for edge in selected_edges:
            weight = random.choice(existing_weights)
            aug_adj[edge[0], edge[1]] = weight
            aug_adj[edge[1], edge[0]] = weight
    return aug_adj

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
subgraph
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


# def aug_subgraph_list(graph_list, drop_percent):
#     graph_num = len(graph_list)
#     aug_list = []
#     for i in range(graph_num):
#         s_graph = aug_subgraph(graph_list[i], drop_percent)
#         aug_list.append(s_graph)
#     return aug_list
#
#
# def aug_subgraph(graph, drop_percent):
#     graph = copy.deepcopy(graph)
#     num = graph.number_of_nodes()
#     all_node_list = [i for i in range(num)]
#     s_num = int(num * (1 - drop_percent))
#     center_node_id = random.randint(0, num - 1)
#     sub_node_id_list = [center_node_id]
#     all_neighbor_list = []
#     for i in range(s_num - 1):
#
#         all_neighbor_list += graph.successors(sub_node_id_list[i]).numpy().tolist()
#         all_neighbor_list = list(set(all_neighbor_list))
#         new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
#         if len(new_neighbor_list) != 0:
#             new_node = random.sample(new_neighbor_list, 1)[0]
#             sub_node_id_list.append(new_node)
#         else:
#             break
#     del_node_list = [i for i in all_node_list if not i in sub_node_id_list]
#     graph.remove_nodes(del_node_list)
#     return graph
#
#
#
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# add links
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#
# def aug_add_edges(graph, drop_percent=0.2):
#
#     aug_graph = copy.deepcopy(graph)
#     edge_num = aug_graph.number_of_edges()
#     add_num = int(edge_num * drop_percent / 2)
#
#     node_num = aug_graph.number_of_nodes()
#     l = []
#     for i in range(node_num):
#         for j in range(i):
#             l.append((i, j))
#     d = random.sample(l, add_num)
#
#     add_edges_src_list = []
#     add_edges_dst_list = []
#
#     for i in range(add_num):
#         if not aug_graph.has_edges_between(d[i][0], d[i][1]):
#             add_edges_src_list.append(d[i][0])
#             add_edges_src_list.append(d[i][1])
#             add_edges_dst_list.append(d[i][1])
#             add_edges_dst_list.append(d[i][0])
#     aug_graph.add_edges(add_edges_src_list, add_edges_dst_list)
#
#     return aug_graph
#
#
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# drop links
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#
#
# def aug_drop_edges(graph, drop_percent=0.2):
#     aug_graph = copy.deepcopy(graph)
#     edge_num = aug_graph.number_of_edges()
#     drop_num = int(edge_num * drop_percent / 2)
#
#     del_edges_id_list = []
#     all_edges_id_list = [i for i in range(edge_num)]
#     for i in range(drop_num):
#         random_idx = randint(0, edge_num - 1)
#         u_v = aug_graph.find_edges(all_edges_id_list[random_idx])
#         del_edge_id1 = aug_graph.edge_ids(u_v[0], u_v[1])
#         # del_edge_id2 = aug_graph.edge_ids(u_v[1], u_v[0])
#         del_edges_id_list.append(del_edge_id1)
#         # del_edges_id_list.append(del_edge_id2)
#         all_edges_id_list.remove(del_edge_id1.item())
#         # all_edges_id_list.remove(del_edge_id2.item())
#         edge_num -= 1
#
#     aug_graph.remove_edges(del_edges_id_list)
#
#     return aug_graph
#
#
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# drop nodes
# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#
#
# def aug_drop_node_list(graph_list, drop_percent):
#     graph_num = len(graph_list)  # number of graphs
#     aug_list = []
#     for i in range(graph_num):
#         aug_graph = aug_drop_node(graph_list[i], drop_percent)
#         aug_list.append(aug_graph)
#     return aug_list
#
#
# def aug_drop_node(graph, drop_percent=0.2):
#     num = graph.number_of_nodes()  # number of nodes of one graph
#     drop_num = int(num * drop_percent)  # number of drop nodes
#     aug_graph = copy.deepcopy(graph)
#     all_node_list = [i for i in range(num)]
#     drop_node_list = random.sample(all_node_list, drop_num)
#     edges_to_remove = []
#     for node in drop_node_list:
#         # 获取离开该节点的所有边的ID（适用于有向图）
#         out_edge_ids = aug_graph.out_edges(node, form='eid')
#         # 将这些边添加到待删除的边列表中
#         for edge_id in out_edge_ids:
#             edges_to_remove.append(edge_id)
#
#     # 删除这些边
#     aug_graph.remove_edges(edges_to_remove)
#     return aug_graph