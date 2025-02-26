import os
import torch
import random
import logging
import numpy as np


# 1.logger file
def log_string(log, string):
    """
    :param log: file pointer
    :param string: string to write to file
    """
    log.write(string + '\n')
    log.flush()
    print(string)


# 日志流对象
def get_logger(root, name=None, debug=True):
    logger = logging.getLogger(name)

    if not logger.handlers:  # 只有在没有处理器的情况下才添加处理器
        logger.setLevel(logging.DEBUG)

        # define the formate
        formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
        # create another handler for output log to console
        console_handler = logging.StreamHandler()
        if debug:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
            logfile = os.path.join(root, 'run.log')
            print('Creat Log File in: ', logfile)
            file_handler = logging.FileHandler(logfile, mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)  # 只有在非调试模式下才添加文件处理器
        console_handler.setFormatter(formatter)
        # add Handler to logger
        logger.addHandler(console_handler)
        # if not debug:
        #     logger.addHandler(file_handler)
    return logger


# 2.Initialize the random number seed
def init_seed(seed):
    """
    Disable cudnn to maximize reproducibility
    """
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_device(opt):
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt


def init_optim(model, opt):
    # Initialize the optimizer
    return torch.optim.Adam(params=model.parameters(), lr=opt.lr_init)


def init_lr_scheduler(optim, opt):
    # Initialize the learning strategy
    # return torch.optim.lr_scheduler.StepLR(optimizer=optim,gamma=opt.lr_scheduler_rate,step_size=opt.lr_scheduler_step)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=opt.lr_decay_steps,
                                                gamma=opt.lr_scheduler_rate)


def print_model_parameters(model, only_num=True):
    # Record the trainable parameters of the model
    if not only_num:
        for name, param in model.named_parameters():
            continue
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Model Trainable Parameters: {:,}'.format(total_num))


def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 * 1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024 * 1024.)
    print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))
    return allocated_memory, cached_memory


import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4):
        super(AttentionGatingNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 特征提取层
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 多头注意力层
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # 输出层
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        x: 输入数据，形状为 (B, T, N, D)
        """
        B, T, N, D = x.size()

        # 聚合时间和节点维度
        x = x.view(B, T * N, D)  # (B, T*N, D)

        # 特征提取
        x = F.relu(self.fc1(x))  # (B, T*N, hidden_dim)

        # 注意力机制
        attn_output, attn_weights = self.attention(x, x, x)  # (B, T*N, hidden_dim)

        # 全局池化（平均）
        attn_output = attn_output.mean(dim=1)  # (B, hidden_dim)

        # 输出门控值
        gate = self.activation(self.fc_out(attn_output))  # (B, 1)

        return gate


class VarianceAlphaAdjuster:
    def __init__(self, alpha_min=0.3, alpha_max=0.7):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.prev_variance = None  # 保存上一批次的方差
        self.max_variance_change = 0  # 记录方差变化的最大值
        self.min_variance_change = float('inf')  # 记录方差变化的最小值

    def update_alpha(self, current_output):
        # 计算当前批次的方差
        current_variance = current_output.var().item()

        # 若是第一批次，没有上一批次的方差，则跳过计算
        if self.prev_variance is None:
            self.prev_variance = current_variance
            return self.alpha_min

        # 计算当前方差与上一批次方差的差异
        variance_change = abs(current_variance - self.prev_variance)

        # 更新最大和最小方差变化值
        self.max_variance_change = max(self.max_variance_change, variance_change)
        self.min_variance_change = min(self.min_variance_change, variance_change)

        # 更新 prev_variance 为当前方差
        self.prev_variance = current_variance

        # 防止除零错误
        if self.max_variance_change == self.min_variance_change:
            normalized_alpha = self.alpha_min
        else:
            # 归一化 variance_change 到 [0, 1] 区间
            normalized_change = (variance_change - self.min_variance_change) / (
                        self.max_variance_change - self.min_variance_change)

            # 将归一化后的变化映射到 [alpha_min, alpha_max] 区间
            normalized_alpha = self.alpha_min + normalized_change * (self.alpha_max - self.alpha_min)

        return normalized_alpha


# 使用EMA更新教师模型参数
def update_teacher_model(teacher_model, student_model, alpha):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(1 - alpha).add_(student_param.data, alpha=alpha)



# SynFlow得分计算函数
def get_synflow_scores(model, device):
    scores = {}
    signs = linearize_model(model)  # 线性化模型
    for name, param in model.named_parameters():
        if param.grad is not None:
            scores[name] = torch.clone(param.grad * param).detach().abs_()
            param.grad.zero_()
    nonlinearize_model(model, signs)  # 还原模型
    return scores

# 线性化和非线性化模型的辅助函数
@torch.no_grad()
def linearize_model(model):
    signs = {}
    for name, param in model.state_dict().items():
        signs[name] = torch.sign(param)
        param.abs_()
    return signs

@torch.no_grad()
def nonlinearize_model(model, signs):
    for name, param in model.state_dict().items():
        param.mul_(signs[name])
