import argparse
import pickle
from random import random

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
import copy

sys.path.append('../')
# from lib.ST_aug_copy import *
from lib.ST_aug import *
from generate_adj_mx import *
from lib.utils import *


VarianceAlphaAdjuster = VarianceAlphaAdjuster(alpha_min=0.4, alpha_max=0.6)

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data




def set_env(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


sys.path.append("..")
from lib.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.data_loader import *
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from model.STAEformer import STAEformer
from lib.ST_aug import *

# ! X shape: (B, T, N, C)


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(
        model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


def mkd_train_one_epoch(
        model, trainset_loader, optimizer, scheduler, criterion, clip_grad, alpha, model_lambda, teacher_model, stu, adj_mx, SCALER,log=None
):
    global cfg, global_iter_count, global_target_length, DEVICE, VarianceAlphaAdjuster

    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        dynamic_alpha = VarianceAlphaAdjuster.update_alpha(x_batch.cuda())

        # print(dynamic_alpha)
        if dynamic_alpha > model_lambda:
            aug_x = get_aug_data(adj_mx, x_batch)

            aug_output = model(aug_x)
            aug_teacher_model_output = teacher_model(aug_x)

            out_batch_aug = SCALER.inverse_transform(aug_output)
            teacher_batch_aug = SCALER.inverse_transform(aug_teacher_model_output)
            loss = criterion(out_batch_aug, teacher_batch_aug)
        else:
            out_batch = model(x_batch)
            out_batch = SCALER.inverse_transform(out_batch)
            loss = criterion(out_batch, y_batch)


        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        update_teacher_model(teacher_model, model, alpha)




    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


def train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        clip_grad=0,
        max_epochs=200,
        early_stop=10,
        verbose=1,
        plot=False,
        log=None,
        save=None,
):
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []
    train_time = []
    inference_time = []

    for epoch in range(max_epochs):
        t1 = time.time()
        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log
        )
        t2 = time.time()
        val_loss = eval_model(model, valset_loader, criterion)
        t3 = time.time()
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_time.append(t2 - t1)
        inference_time.append(t3 - t2)
        if (epoch + 1) % verbose == 0:
            print(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                "Training Time: %.4f secs" % (t2 - t1),
                "Inference Time: %.4f secs" % (t3 - t2),
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch

            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, save)
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    # out_str = f"Early stopping at epoch: {epoch + 1}\n"
    # out_str += f"Best at epoch {best_epoch + 1}:\n"
    # out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    # out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
    #     train_rmse,
    #     train_mae,
    #     train_mape,
    # )
    # out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    # out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
    #     val_rmse,
    #     val_mae,
    #     val_mape,
    # )
    # print_log(out_str, log=log)
    print_log("Mean training time: {:.4f} s, Mean inference time: {:.4f} s".format(np.mean(train_time),
                                                                                   np.mean(inference_time)),
              log=log)
    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    return model

def are_models_dict_equal(student_state_dict, teacher_state_dict):


    # 比较两个状态字典的所有键和值
    for key in student_state_dict:
        if key not in teacher_state_dict:
            print(f"Key {key} not found in teacher model")
            return False
        if not torch.equal(student_state_dict[key], teacher_state_dict[key]):
            print(f"Mismatch found at {key}")
            return False

    return True


def are_models_equal(student_model, teacher_model):
    student_state_dict = student_model.state_dict()
    teacher_state_dict = teacher_model.state_dict()

    # 比较两个状态字典的所有键和值
    for key in student_state_dict:
        if key not in teacher_state_dict:
            print(f"Key {key} not found in teacher model")
            return False
        if not torch.equal(student_state_dict[key], teacher_state_dict[key]):
            print(f"Mismatch found at {key}")
            return False

    return True

def mkd_train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        alpha,
        model_lambda,
        teacher_model,
        stu,
        adj_mx,
        SCALER,
        save_model,
        clip_grad=0,
        max_epochs=200,
        early_stop=10,
        verbose=1,
        plot=False,
        log=None,
        save=None,
):
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []
    train_time = []
    inference_time = []
    # 构建KD树
    build_kd_tree(adj_mx)

    # 冻结
    frozen_parameters = set()
    total_steps = max_epochs // 10  # 总冻结步骤数 N
    cumulative_frozen = 0  # 累计冻结的参数数量
    P = None  # 总参数数量，将在第一次冻结时初始化
    P_reserve = None  # 可冻结的总参数数量 (P - R)
    step_freeze = 0  # 当前冻结步骤
    reserve_params = 1  # 保留参数数量 R
    sum_weights = total_steps * (total_steps + 1) // 2  # sum_weights = N*(N+1)/2


    for epoch in range(1, max_epochs+1, 1):
        # if epoch % 10 == 0:
        #     # 获取 SynFlow 得分
        #     scores = get_synflow_scores(model, DEVICE)
        #
        #     if P is None:
        #         P = len(scores)
        #         P_reserve = P - reserve_params
        #         print(f"总参数数量: {P}")
        #         print(f"保留未冻结参数数量: {reserve_params}")
        #         print(f"可冻结参数总数: {P_reserve}")
        #         print(f"总冻结步骤数: {total_steps}")
        #         print(f"sum_weights: {sum_weights}")
        #
        #     step_freeze += 1  # 当前步骤增加
        #
        #     k = step_freeze  # 当前冻结步骤，从1到N
        #     N = total_steps
        #
        #     if k <= N:
        #         # 计算每次冻结的参数数量，使用 P_reserve 和 sum_weights
        #         freeze_k = int(round((N - k + 1) * P_reserve / sum_weights))
        #     else:
        #         # 超出总步骤数，防止冻结数量溢出
        #         freeze_k = 0
        #
        #     # 确保冻结数量不超过剩余可冻结的参数数量
        #     remaining_freezable = P_reserve - cumulative_frozen
        #     freeze_k = min(freeze_k, remaining_freezable)
        #
        #     if freeze_k <= 0:
        #         print(f"Epoch {epoch}: 没有更多参数可以冻结。")
        #         continue
        #
        #     # 根据每个张量得分的平均值进行排序，并选择得分最高的参数
        #     sorted_params = sorted(
        #         [(name, score) for name, score in scores.items() if name not in frozen_parameters],
        #         key=lambda x: x[1].mean() if isinstance(x[1], torch.Tensor) else x[1],
        #         reverse=False
        #     )
        #
        #     params_to_freeze = sorted_params[:freeze_k]
        #
        #     # 冻结选择的参数
        #     for name, _ in params_to_freeze:
        #         param = dict(model.named_parameters())[name]
        #         param.requires_grad = False  # 冻结参数
        #         frozen_parameters.add(name)  # 添加到已冻结列表
        #
        #     cumulative_frozen += freeze_k
        #
        #     print(
        #         f"Epoch {epoch}: 冻结了 {freeze_k} / {P} 个参数，总冻结参数数量为 {cumulative_frozen} / {P_reserve}。")
        #
        #     # 如果所有可冻结参数已被冻结，打印信息
        #     if cumulative_frozen >= P_reserve:
        #         print(f"所有可冻结的参数已被冻结。保留的参数数量为 {reserve_params}。")

        t1 = time.time()
        train_loss = mkd_train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, alpha, model_lambda, teacher_model, stu, adj_mx, SCALER,
            log=log
        )
        t2 = time.time()
        val_loss = eval_model(model, valset_loader, criterion)
        t3 = time.time()
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_time.append(t2 - t1)
        inference_time.append(t3 - t2)
        if (epoch + 1) % verbose == 0:
            print(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                "Training Time: %.4f secs" % (t2 - t1),
                "Inference Time: %.4f secs" % (t3 - t2),
            )


        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            if save_model == 1:
                teacher_parameters = copy.deepcopy(teacher_model.state_dict())
                for key in best_state_dict:
                    if 'mask' in key:
                        continue
                    best_state_dict[key] = (best_state_dict[key] + teacher_parameters[key]) / 2
            torch.save(best_state_dict, save)
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)
    # train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    # val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))
    #
    # out_str = f"Early stopping at epoch: {epoch + 1}\n"
    # out_str += f"Best at epoch {best_epoch + 1}:\n"
    # out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    # out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
    #     train_rmse,
    #     train_mae,
    #     train_mape,
    # )
    # out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    # out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
    #     val_rmse,
    #     val_mae,
    #     val_mape,
    # )
    # print_log(out_str, log=log)
    print_log("Mean training time: {:.4f} s, Mean inference time: {:.4f} s".format(np.mean(train_time),
                                                                                   np.mean(inference_time)),
              log=log)
    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    # if save:
    #     torch.save(best_state_dict, save)
    return model


def float_range(start, stop, step):
    while start <= stop:
        yield round(start, 10)  # 使用 round() 函数来控制浮点数精度
        start += step


@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    # print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        if i + 1 in [3, 6, 12]:
            rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
            out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
                i + 1,
                rmse,
                mae,
                mape,
            )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    mode = "fine"

    # -------------------------- set running environment ------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems08")
    parser.add_argument("-g", "--gpu_num", type=int, default=1)
    parser.add_argument("-m", "--mode", type=str, default=0)
    args = parser.parse_args()


    # seed = torch.randint(1000, (1,))  # set random seed here
    seed_everything(42)
    # set_cpu_num(1)
    # GPU_ID = args.gpu_num
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = STAEformer.__name__

    with open(f"{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #

    model = STAEformer(**cfg["model_args"])

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}--{mode}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    # print_log(dataset, log=log)
    # (
    #     trainset_loader,
    #     valset_loader,
    #     testset_loader,
    #     SCALER,
    # ) = get_dataloaders_from_index_data(
    #     -1,
    #     cfg["train_size"],
    #     cfg["val_size"],
    #     data_path,
    #     tod=cfg.get("time_of_day"),
    #     dow=cfg.get("day_of_week"),
    #     batch_size=cfg.get("batch_size", 64),
    #     log=log,
    # )
    # print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    if dataset in ("METRLA", "PEMSBAY"):
        criterion = MaskedMAELoss()
    elif dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        criterion = nn.HuberLoss()
    else:
        raise ValueError("Unsupported dataset.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # --------------------------- print model structure -------------------------- #

    # print_log("---------", model_name, "---------", log=log)
    # print_log(
    #     json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    # )
    # print_log(
    #     summary(
    #         model,
    #         [
    #             cfg["batch_size"],
    #             cfg["in_steps"],
    #             cfg["num_nodes"],
    #             next(iter(trainset_loader))[0].shape[-1],
    #         ],
    #         verbose=0,  # avoid print twice
    #     ),
    #     log=log,
    # )
    # print_log(log=log)

    # --------------------------- train and test model --------------------------- #
    if dataset in ("METRLA"):
        graph_pkl = "../data/METRLA/adj_mx.pkl"
        _, _, adj_mx = load_pickle(graph_pkl)

    if dataset in ("PEMSBAY"):
        graph_pkl = "../data/PEMSBAY/adj_mx.pkl"
        _, _, adj_mx = load_pickle(graph_pkl)

    if dataset in ("PEMS04"):
        GRAPH = "../data/PEMS04/PEMSD4.csv"
        adj_mx = get_adjacency_matrix(GRAPH, cfg.get("num_nodes"), type='connectivity', id_filename=None)

    if dataset in ("PEMS08"):
        GRAPH = "../data/PEMS08/PEMSD8.csv"
        adj_mx = get_adjacency_matrix(GRAPH, cfg.get("num_nodes"), type='connectivity', id_filename=None)

    print_log(f"Loss: {criterion._get_name()}", log=log)

    if mode == "one":
        trainset_loader, valset_loader, testset_loader, SCALER = get_dataloaders_from_index_data(
            1,
            cfg["train_size"],
            cfg["val_size"],
            data_path,
            tod=cfg.get("time_of_day"),
            dow=cfg.get("day_of_week"),
            batch_size=cfg.get("batch_size", 64),
            log=log,
        )
        save_path = f"../saved_models/fine/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save = os.path.join(save_path, f"{model_name}-{dataset}-fine_1.pt")

        # train(
        #     model,
        #     trainset_loader,
        #     valset_loader,
        #     optimizer,
        #     scheduler,
        #     criterion,
        #     clip_grad=cfg.get("clip_grad"),
        #     max_epochs=cfg.get("max_epochs", 100),
        #     early_stop=cfg.get("early_stop", 10),
        #     verbose=1,
        #     log=log,
        #     save=save,
        # )

        # load model and test one
        model.load_state_dict(torch.load(save))
        model = model.to(DEVICE)

        for index in range(2, 6):
            seed_everything(42)

            print_log(f"one_{index}", log=log)
            trainset_loader, valset_loader, testset_loader, SCALER = get_dataloaders_from_index_data(
                index,
                cfg["train_size"],
                cfg["val_size"],
                data_path,
                tod=cfg.get("time_of_day"),
                dow=cfg.get("day_of_week"),
                batch_size=cfg.get("batch_size", 64),
                log=log,
            )
            test_model(model, testset_loader, log=log)


    elif mode == "fine":
        for index in range(2, 6):
            seed_everything(42)
            print_log(f"fine {index}", log=log)
            trainset_loader, valset_loader, testset_loader, SCALER = get_dataloaders_from_index_data(
                index,
                cfg["train_size"],
                cfg["val_size"],
                data_path,
                tod=cfg.get("time_of_day"),
                dow=cfg.get("day_of_week"),
                batch_size=cfg.get("batch_size", 64),
                log=log,
            )
            save_path = f"../saved_models/fine/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if index != 1:
                if index == 2:
                    save = os.path.join(save_path, f"{model_name}-{dataset}-fine_{index - 1}.pt")
                else:
                    save = os.path.join(save_path, f"{model_name}-{dataset}-fine_{index - 1}.pt")
                model.load_state_dict(torch.load(save))

            save = os.path.join(save_path, f"{model_name}-{dataset}-fine_{index}.pt")
            train(
                model,
                trainset_loader,
                valset_loader,
                optimizer,
                scheduler,
                criterion,
                clip_grad=cfg.get("clip_grad"),
                max_epochs=cfg.get("max_epochs", 100),
                early_stop=cfg.get("early_stop", 10),
                verbose=1,
                log=log,
                save=save,
            )

            model.load_state_dict(torch.load(save))
            test_model(model, testset_loader, log=log)

    elif mode == "mkd":
        for alpha in [0.01]:
            # for model_lambda in float_range(0.3, 0.7, 0.2):
            for model_lambda in [0.55]:
                for stu in [2]:
                    if stu == 2:
                        save_model = 1
                    else:
                        save_model = 1
                    for index in range(2, 6):
                        seed_everything(42)

                        print_log(f"mkd {index} {alpha} {model_lambda} {stu} {save_model}", log=log)
                        trainset_loader, valset_loader, testset_loader, SCALER = get_dataloaders_from_index_data(
                            index,
                            cfg["train_size"],
                            cfg["val_size"],
                            data_path,
                            tod=cfg.get("time_of_day"),
                            dow=cfg.get("day_of_week"),
                            batch_size=cfg.get("batch_size", 64),
                            log=log,
                        )
                        save_path = f"../saved_models/mkd/"
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        if index != 2:
                            save = os.path.join(save_path,
                                                f"{model_name}-{dataset}-mkd_{alpha}_{model_lambda}_{index - 1}_{stu}_{save_model}.pt")
                            model.load_state_dict(torch.load(save))
                        else:
                            tem_save_path = f"../saved_models/fine/"
                            save = os.path.join(tem_save_path, f"{model_name}-{dataset}-fine_1.pt")
                            model.load_state_dict(torch.load(save))

                        teacher_model = copy.deepcopy(model).to(DEVICE)

                        save = os.path.join(save_path, f"{model_name}-{dataset}-mkd_{alpha}_{model_lambda}_{index}_{stu}_{save_model}.pt")
                        mkd_train(
                            model,
                            trainset_loader,
                            valset_loader,
                            optimizer,
                            scheduler,
                            criterion,
                            alpha,
                            model_lambda,
                            teacher_model,
                            stu,
                            adj_mx,
                            SCALER,
                            save_model,
                            clip_grad=cfg.get("clip_grad"),
                            max_epochs=cfg.get("max_epochs", 100),
                            early_stop=cfg.get("early_stop", 10),
                            verbose=1,
                            log=log,
                            save=save,


                        )
                        model.load_state_dict(torch.load(save))
                        test_model(model, testset_loader, log=log)

    # print_log(f"Saved Model: {save}", log=log)

    log.close()
