import numpy as np
import torch.nn.functional as F
def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)

def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


import torch


# def get_synflow_score(model, dataloader, device):
#     """
#     获取模型参数的 SynFlow 得分。
#
#     参数:
#     - model: 需要计算得分的神经网络模型。
#     - dataloader: 数据加载器，用于提供单个批次的数据。
#     - device: 计算设备，例如 "cpu" 或 "cuda"。
#
#     返回:
#     - scores: 包含每个参数的 SynFlow 得分的字典。
#     """
#
#     # 定义线性化和非线性化模型的辅助函数
#     @torch.no_grad()
#     def linearize(model):
#         signs = {}
#         for name, param in model.state_dict().items():
#             signs[name] = torch.sign(param)
#             param.abs_()
#         return signs
#
#     @torch.no_grad()
#     def nonlinearize(model, signs):
#         for name, param in model.state_dict().items():
#             param.mul_(signs[name])
#
#     # 线性化模型
#     signs = linearize(model)
#
#     # 获取数据和目标，确保在计算设备上
#     data, target = next(iter(dataloader))
#     data, target = data.to(device), target.to(device)
#
#     # 前向传播和反向传播
#     output = model(data)
#     torch.sum(output).backward()
#
#     # 计算每个参数的 SynFlow 得分
#     scores = {}
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             scores[name] = torch.clone(param.grad * param).detach().abs_()
#             param.grad.zero_()
#
#     # 还原模型到非线性状态
#     nonlinearize(model, signs)
#
#     return scores


#
# # SynFlow得分计算函数
# def get_synflow_scores(model, device):
#     scores = {}
#     signs = linearize_model(model)  # 线性化模型
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             scores[name] = torch.clone(param.grad * param).detach().abs_()
#             param.grad.zero_()
#     nonlinearize_model(model, signs)  # 还原模型
#     return scores
#
# # 线性化和非线性化模型的辅助函数
# @torch.no_grad()
# def linearize_model(model):
#     signs = {}
#     for name, param in model.state_dict().items():
#         signs[name] = torch.sign(param)
#         param.abs_()
#     return signs
#
# @torch.no_grad()
# def nonlinearize_model(model, signs):
#     for name, param in model.state_dict().items():
#         param.mul_(signs[name])


class KLAlphaAdjuster:
    def __init__(self, alpha_min=0.4, alpha_max=0.6):
        self.alpha_min = alpha_min  # alpha 的最小值
        self.alpha_max = alpha_max  # alpha 的最大值
        self.kl_min = float('inf')  # 初始 KL 散度的最小值
        self.kl_max = float('-inf')  # 初始 KL 散度的最大值
        self.prev_aug_distribution = None  # 保存上一批次增强数据的分布

    def compute_kl_divergence(self, p, q):
        """
        计算 KL 散度 D_KL(P || Q)
        假设 p 和 q 是两个分布的概率向量
        """
        kl_div = F.kl_div(q.log(), p, reduction="batchmean")
        return kl_div.item()

    def update_alpha(self, current_aug_distribution):
        """
        根据增强数据 KL 散度更新 alpha 值，比较当前增强数据分布和上一批次增强数据分布
        """
        if self.prev_aug_distribution is None:
            # 若是第一批次，没有上一批次增强分布，则跳过KL散度计算
            self.prev_aug_distribution = current_aug_distribution
            return self.alpha_min

        # 计算当前增强数据分布与上一分布之间的 KL 散度
        kl_divergence = self.compute_kl_divergence(self.prev_aug_distribution, current_aug_distribution)
        # 更新 KL 散度的最大值和最小值
        self.kl_min = min(self.kl_min, kl_divergence)
        self.kl_max = max(self.kl_max, kl_divergence)

        # 防止最大值和最小值相等，避免除零错误
        if self.kl_max == self.kl_min:
            normalized_alpha = self.alpha_min
        else:
            # 归一化 KL 散度到 [0, 1]
            normalized_kl = (kl_divergence - self.kl_min) / (self.kl_max - self.kl_min)
            # 将归一化后的 KL 散度映射到 [alpha_min, alpha_max]
            normalized_alpha = self.alpha_min + normalized_kl * (self.alpha_max - self.alpha_min)

        # 更新 prev_aug_distribution 为当前增强分布，用于下次计算
        self.prev_aug_distribution = current_aug_distribution

        return normalized_alpha


from scipy.stats import wasserstein_distance

class WassersteinAlphaAdjuster:
    def __init__(self, alpha_min=0.4, alpha_max=0.6):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.prev_output = None  # 保存上一批次的输出分布

    def update_alpha(self, current_output):
        # 若是第一批次，没有上一批次增强分布，则跳过 Wasserstein 距离计算
        if self.prev_output is None:
            self.prev_output = current_output
            return self.alpha_min

        # 计算 Wasserstein 距离
        w_distance = wasserstein_distance(self.prev_output.flatten().cpu().numpy(),
                                          current_output.flatten().cpu().numpy())

        # 更新 prev_output 为当前分布
        self.prev_output = current_output

        # 根据 Wasserstein 距离归一化到 [alpha_min, alpha_max] 区间
        normalized_alpha = max(self.alpha_min, min(self.alpha_max, self.alpha_min + w_distance * 0.1))

        return normalized_alpha

# 使用EMA更新教师模型参数
def update_teacher_model(teacher_model, student_model, alpha):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(1 - alpha).add_(student_param.data, alpha=alpha)


