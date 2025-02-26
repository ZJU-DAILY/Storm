import sys

sys.path.append('../')

import os
import copy
import time
import torch
from lib.utils import *
from tqdm import tqdm
from lib.evaluate import All_Metrics
from lib.data_loader import *
from lib.ST_aug_copy import *
from lib.ST_aug import *
from lib.data_loader import *
from STGCN_Utils import *

import copy





class Trainer(object):
    def __init__(self, args, train_loader, val_loader, test_loader, scaler, model, loss, optimizer, lr_scheduler,
                 cl=True,
                 new_training_method=True,
                 freeze=True):
        super(Trainer, self).__init__()
        init_seed(args.seed)
        self.args = args
        self.train_loader = train_loader
        self.train_per_epoch = len(train_loader)
        self.val_loader = val_loader
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)
        else:
            self.val_loader = test_loader
            self.val_per_epoch = len(self.val_loader)
        self.test_loader = test_loader
        self.scaler = scaler
        # model, loss_func, optimizer, lr_scheduler
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.freeze = freeze
        # 日志与模型的保存路径
        self.best_path = os.path.join(args.log_dir, '{}_{}_best_model.pth'.format(args.dataset, args.model))
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)  # run.log
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        # self.logger.info("Experiment log path in: {}".format(args.log_dir))
        # self.logger.info(args)

        self.frozen_parameters = set()
        self.total_steps = self.args.epochs // 10  # 总冻结步骤数 N
        self.cumulative_frozen = 0  # 累计冻结的参数数量
        self.P = None  # 总参数数量，将在第一次冻结时初始化
        self.P_reserve = None  # 可冻结的总参数数量 (P - R)
        self.step_freeze = 0  # 当前冻结步骤
        self.reserve_params = 1  # 保留参数数量 R
        self.sum_weights = self.total_steps * (self.total_steps + 1) // 2  # sum_weights = N*(N+1)/2
        self.VarianceAlphaAdjuster = VarianceAlphaAdjuster(alpha_min=0.4, alpha_max=0.6)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for _, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]
            # data and target shape: B, T, N, D; output shape: B, T, N, D
            self.optimizer.zero_grad()
            # output = self.model(data)
            output = self.model(data)  # directly predict the true value
            # if self.args.real_value:
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)  # 若模型预测的真实值
            loss = self.loss(output.cuda(), label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()

        train_epoch_loss = total_loss / self.train_per_epoch
        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss



    def mkd_train_epoch(self, teacher_model, alpha, model_lambda, adj_mx):
        self.model.train()
        total_loss = 0
        # 在更新教师模型后
        # for param_t, param_s in zip(teacher_model.parameters(), self.model.parameters()):
        #     print(torch.sum(param_t - param_s).item())
        #     break  # 只打印第一个参数的变化，避免信息过多
        for _, (data, target) in enumerate(self.train_loader):

            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]

            # data and target shape: B, T, N, D; output shape: B, T, N, D
            self.optimizer.zero_grad()

            if self.args.real_value:
                label = self.scaler.inverse_transform(label)  # 若模型预测的真实值
            # 根据增强数据的上一次和当前分布的 KL 散度动态调整 alpha

            dynamic_alpha = self.VarianceAlphaAdjuster.update_alpha(data.cuda())

            # print(dynamic_alpha)
            if dynamic_alpha > model_lambda:
                aug_x = get_aug_data(adj_mx, data)
                aug_output = self.model(aug_x)  # directly predict the true value
                aug_teacher_model_output = teacher_model(aug_x)  # directly predict the true value
                loss = self.loss(aug_output.cuda(), aug_teacher_model_output.cuda())
            else:
                output = self.model(data)
                loss = self.loss(output.cuda(), label)
            # 如果变化不大 就要使用之前的模型训练 也就是减少原来的LOSS（LOSS是惩罚 ，减小 LOSS2的LOSS，目前是正确的）
            # print(dynamic_alpha)
            # print(model_lambda * loss_d1)
            # print(str(loss1.item()) + str(loss2.item()))
            # print(dynamic_alpha)
            # loss = dynamic_alpha * loss2 + (1 - dynamic_alpha) * loss1

            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            # student_parameters = self.model.state_dict()
            # teacher_parameters = teacher_model.state_dict()
            #
            # for key in teacher_parameters:
            #     # mask 不参与更新
            #     if 'mask' in key:
            #         continue
            #     teacher_parameters[key] = alpha * student_parameters[key] + (1.0 - alpha) * teacher_parameters[key]
            #
            # # 将更新后的参数重新设置回teacher_model
            # teacher_model.load_state_dict(teacher_parameters)
            update_teacher_model(teacher_model, self.model, alpha)


        train_epoch_loss = total_loss / self.train_per_epoch
        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        # 在训练循环中更新教师模型

        return train_epoch_loss

    def val_epoch(self):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.val_loader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output = self.model(data)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)  # 若模型预测的真实值
                loss = self.loss(output.cuda(), label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        return val_loss


    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        train_time = []
        inference_time = []


        for epoch in tqdm(range(1, self.args.epochs + 1)):
            # 每隔10个epoch，计算SynFlow得分并冻结部分参数
            if self.freeze is True and epoch % 10 == 0 :
                # 获取 SynFlow 得分
                scores = get_synflow_scores(self.model, self.args.device)

                if self.P is None:
                    self.P = len(scores)
                    self.P_reserve = self.P - self.reserve_params
                    print(f"总参数数量: {self.P}")
                    print(f"保留未冻结参数数量: {self.reserve_params}")
                    print(f"可冻结参数总数: {self.P_reserve}")
                    print(f"总冻结步骤数: {self.total_steps}")
                    print(f"sum_weights: {self.sum_weights}")

                self.step_freeze += 1  # 当前步骤增加

                k = self.step_freeze  # 当前冻结步骤，从1到N
                N = self.total_steps

                if k <= N:
                    # 计算每次冻结的参数数量，使用 P_reserve 和 sum_weights
                    freeze_k = int(round((N - k + 1) * self.P_reserve / self.sum_weights))
                else:
                    # 超出总步骤数，防止冻结数量溢出
                    freeze_k = 0

                # 确保冻结数量不超过剩余可冻结的参数数量
                remaining_freezable = self.P_reserve - self.cumulative_frozen
                freeze_k = min(freeze_k, remaining_freezable)

                if freeze_k <= 0:
                    print(f"Epoch {epoch}: 没有更多参数可以冻结。")
                    continue

                # 根据每个张量得分的平均值进行排序，并选择得分最低的参数
                sorted_params = sorted(
                    [(name, score) for name, score in scores.items() if name not in self.frozen_parameters],
                    key=lambda x: x[1].mean() if isinstance(x[1], torch.Tensor) else x[1],
                    reverse=False
                )


                params_to_freeze = sorted_params[:freeze_k]

                # 冻结选择的参数
                for name, _ in params_to_freeze:
                    param = dict(self.model.named_parameters())[name]
                    param.requires_grad = False  # 冻结参数
                    self.frozen_parameters.add(name)  # 添加到已冻结列表

                self.cumulative_frozen += freeze_k

                print(
                    f"Epoch {epoch}: 冻结了 {freeze_k} / {self.P} 个参数，总冻结参数数量为 {self.cumulative_frozen} / {self.P_reserve}。")

                # 如果所有可冻结参数已被冻结，打印信息
                if self.cumulative_frozen >= self.P_reserve:
                    print(f"所有可冻结的参数已被冻结。保留的参数数量为 {self.reserve_params}。")

            t1 = time.time()
            train_epoch_loss = self.train_epoch()
            t2 = time.time()
            # 验证, 如果是Encoder-Decoder结构，则需要将epoch作为参数传入
            val_epoch_loss = self.val_epoch()
            t3 = time.time()
            train_time.append(t2 - t1)
            inference_time.append(t3 - t2)
            print(
                'Epoch {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.4f} secs, Inference Time: {:.4f} secs.'.format(
                    epoch, train_epoch_loss, val_epoch_loss, (t2 - t1), (t3 - t2)))

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning("Gradient explosion detected. Ending...")
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # is or not early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state

            if best_state == True:
                # self.logger.info("Current best model saved!")
                best_model = copy.deepcopy(self.model.state_dict())


                torch.save(best_model, self.best_path)

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f} min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # save the best model to file
        # self.logger.info("Saving current best model to " + self.best_path)
        # load model and test
        self.logger.info("Mean training time: {:.4f} s, Mean inference time: {:.4f} s".format(np.mean(train_time),
                                                                                              np.mean(inference_time)))
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def mkd_train(self, teacher_model, alpha, model_lambda, adj_mx, save_model):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        train_time = []
        inference_time = []
        # 构建KD树
        build_kd_tree(adj_mx.cpu())
        for epoch in tqdm(range(1, self.args.epochs + 1)):
            if self.freeze is True and epoch % 10 == 0 :
                # 获取 SynFlow 得分
                scores = get_synflow_scores(self.model, self.args.device)

                if self.P is None:
                    self.P = len(scores)
                    self.P_reserve = self.P - self.reserve_params
                    print(f"总参数数量: {self.P}")
                    print(f"保留未冻结参数数量: {self.reserve_params}")
                    print(f"可冻结参数总数: {self.P_reserve}")
                    print(f"总冻结步骤数: {self.total_steps}")
                    print(f"sum_weights: {self.sum_weights}")

                self.step_freeze += 1  # 当前步骤增加

                k = self.step_freeze  # 当前冻结步骤，从1到N
                N = self.total_steps

                if k <= N:
                    # 计算每次冻结的参数数量，使用 P_reserve 和 sum_weights
                    freeze_k = int(round((N - k + 1) * self.P_reserve / self.sum_weights))
                else:
                    # 超出总步骤数，防止冻结数量溢出
                    freeze_k = 0

                # 确保冻结数量不超过剩余可冻结的参数数量
                remaining_freezable = self.P_reserve - self.cumulative_frozen
                freeze_k = min(freeze_k, remaining_freezable)

                if freeze_k <= 0:
                    print(f"Epoch {epoch}: 没有更多参数可以冻结。")
                    continue

                # 根据每个张量得分的平均值进行排序，并选择得分最高的参数
                sorted_params = sorted(
                    [(name, score) for name, score in scores.items() if name not in self.frozen_parameters],
                    key=lambda x: x[1].mean() if isinstance(x[1], torch.Tensor) else x[1],
                    reverse=False
                )

                params_to_freeze = sorted_params[:freeze_k]

                # 冻结选择的参数
                for name, _ in params_to_freeze:
                    param = dict(self.model.named_parameters())[name]
                    param.requires_grad = False  # 冻结参数
                    self.frozen_parameters.add(name)  # 添加到已冻结列表

                self.cumulative_frozen += freeze_k

                print(
                    f"Epoch {epoch}: 冻结了 {freeze_k} / {self.P} 个参数，总冻结参数数量为 {self.cumulative_frozen} / {self.P_reserve}。")

                # 如果所有可冻结参数已被冻结，打印信息
                if self.cumulative_frozen >= self.P_reserve:
                    print(f"所有可冻结的参数已被冻结。保留的参数数量为 {self.reserve_params}。")

            t1 = time.time()
            train_epoch_loss = self.mkd_train_epoch(teacher_model, alpha, model_lambda, adj_mx)
            t2 = time.time()
            # 验证, 如果是Encoder-Decoder结构，则需要将epoch作为参数传入
            val_epoch_loss = self.val_epoch()
            t3 = time.time()
            train_time.append(t2 - t1)
            inference_time.append(t3 - t2)
            print(
                'Epoch {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.4f} secs, Inference Time: {:.4f} secs.'.format(
                    epoch, train_epoch_loss, val_epoch_loss, (t2 - t1), (t3 - t2)))
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning("Gradient explosion detected. Ending...")
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # is or not early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                # self.logger.info("Current best model saved!")
                best_model = copy.deepcopy(self.model.state_dict())
                if save_model == 1:
                    teacher_parameters = copy.deepcopy(teacher_model.state_dict())
                    for key in best_model:
                        if 'mask' in key:
                            continue
                        best_model[key] = (best_model[key] + teacher_parameters[key]) / 2

                torch.save(best_model, self.best_path)
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f} min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # save the best model to file
        # self.logger.info("Saving current best model to " + self.best_path)
        # load model and test
        self.logger.info("Mean training time: {:.4f} s, Mean inference time: {:.4f} s".format(np.mean(train_time),
                                                                                              np.mean(inference_time)))
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, save_path=None):
        if save_path != None:
            model.load_state_dict(torch.load(save_path))
            model.to(args.device)
            print("load saved model...")
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for _, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                output = model(data)
                y_true.append(label)
                y_pred.append(output)

        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))

        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.mape_thresh)
            if t + 1 in [3, 6, 12]:
                logger.info(
                    "Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        mae, rmse, mape = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))
