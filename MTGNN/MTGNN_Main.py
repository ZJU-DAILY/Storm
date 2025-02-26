import sys
sys.path.append('../')

import os
import torch
import torch.nn as nn
from datetime import datetime
from lib.utils import *
from lib.data_loader import *
from lib.generate_adj_mx import *
from lib.evaluate import MAE_torch
from MTGNN_Config import args
from MTGNN_Trainer import Trainer
from model.MTGNN.prune_mtgnn import gtnet as Network
from light.pruners import load_pruner
from light import generator
from light import do_prune
import light.metric
import copy

def load_data(args, index):
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(index,args,
                                                                            normalizer=args.normalizer,
                                                                            tod=False,
                                                                            dow=False,
                                                                            single=False)
    if args.graph_type == 'BINARY':
        adj_mx = get_adjacency_matrix(args.graph_path, args.num_node, type='connectivity', id_filename=args.filename_id)
    elif args.graph_type == 'DISTANCE':
        adj_mx = get_Gaussian_matrix(args.graph_path, args.num_node, args.normalized_k, id_filename=args.filename_id)
    print("The shape of adjacency matrix : ", adj_mx.shape)
    # adj_mx = torch.FloatTensor(adj_mx).to(args.device)

    import numpy as np
    # 将邻接矩阵保存为txt文件
    np.savetxt("adj_matrix.txt", adj_mx, fmt='%f')

    # 找到非零元素的坐标
    non_zero_coords = np.nonzero(adj_mx)

    # 提取非零元素的值
    non_zero_values = adj_mx[non_zero_coords]

    # 将非零元素的坐标和值保存为txt文件
    with open("non_zero_values.txt", "w") as file:
        for coord, value in zip(zip(non_zero_coords[0], non_zero_coords[1]), non_zero_values):
            file.write(f"({coord[0]}, {coord[1]}): {value}\n")


    return adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler


def generate_model_components(args):
    init_seed(args.seed)

    # 1. model(fixed params)
    model = Network(
        gcn_true=args.gcn_true,
        buildA_true=args.buildA_true,
        gcn_depth=args.gcn_depth,
        num_nodes=args.num_node,
        device=args.device,
        predefined_A=None,
        static_feat=None,
        dropout=args.dropout,
        subgraph_size=args.subgraph_size,
        node_dim=args.node_dim,  # PEMSD8: 2
        dilation_exponential=args.dilation_exponential,
        conv_channels=args.conv_channels,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        seq_length=args.window,
        in_dim=args.input_dim,
        out_dim=args.horizon,
        layers=args.layers,
        propalpha=args.propalpha,
        tanhalpha=args.tanhalpha,
        layer_norm_affline=args.layer_norm_affline
    )
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    # print the number of model parameters
    print_model_parameters(model, only_num=False)
    # 2. loss
    def masked_mae_loss(scaler, mask_value):
        def loss(preds, labels):
            if scaler:
                preds = scaler.inverse_transform(preds)
                labels = scaler.inverse_transform(labels)
            mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
            return mae
        return loss
    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'smoothloss':
        loss = torch.nn.SmoothL1Loss().to(args.device)
    elif args.loss_func == 'huber':
        loss = torch.nn.HuberLoss(delta=1.0).to(args.device)
    else:
        raise ValueError
    # 3. optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=args.weight_decay, amsgrad=False)
    # 4. learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args.lr_decay_rate)
    return model, loss, optimizer, lr_scheduler

def get_log_dir(mode, model, dataset):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')

    current_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))  # /GNN4Traffic/TrafficSpeed
    log_dir = os.path.join(current_dir, 'log', model, dataset, mode, current_time)
    return log_dir




def float_range(start, stop, step):
    while start <= stop:
        yield round(start, 10)  # 使用 round() 函数来控制浮点数精度
        start += step


def prune_model(args, model, trainer, compress):
    if float(compress) == 1.0:
        return

    dataloader = trainer.train_loader

    prune_x = []
    prune_y = []
    for i, (data, target) in enumerate(dataloader):
        prune_x.append(data.cpu())
        prune_y.append(target.cpu())
        if i + 1 >= args.prune_dataset_ratio:
            break

    data_x = np.concatenate(prune_x, axis=-1)
    data_y = np.concatenate(prune_y, axis=-1)

    # 创建一个新的 DataLoader 对象，这就是你的 prune_loader
    prune_loader = data_loader(data_x, data_y, batch_size=args.batch_size)

    pruner = load_pruner(args.pruner)(
        generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual,
                                    args.prune_layernorm))

    sparsity = compress
    do_prune.prune_loop(model, trainer.loss, pruner, prune_loader, args.device, sparsity,
                        args.compression_schedule, args.mask_scope, args.prune_epochs, scaler, args.reinitialize,
                        args.prune_train_mode,
                        args.shuffle, args.invert)

    prune_result = light.metric.summary(model,
                                        pruner.scores,
                                        light.metric.flop(model, [], args.device),
                                        lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual,
                                                                     args.prune_layernorm))

    total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
    possible_params = prune_result['size'].sum()
    # total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
    # possible_flops = prune_result['flops'].sum()

    trainer.logger.info(
        "Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
    # trainer.logger.info("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'
    init_seed(args.seed)
    current_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))  # /GNN4Traffic/TrafficSpeed
    model_dir = os.path.join(current_dir, 'save', args.model, args.dataset)
    os.makedirs(model_dir, exist_ok=True)

    args.log_dir = get_log_dir(args.mode, args.model, args.dataset)

    if os.path.isdir(args.log_dir) == False and not args.debug:
        os.makedirs(args.log_dir, exist_ok=True)  # run.log

    if args.mode == "one":
        adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args, 1)

        model, loss, optimizer, lr_scheduler = generate_model_components(args)
        trainer = Trainer(
            args=args,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            test_loader=test_dataloader,
            scaler=scaler,
            model=model,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )
        save_model_dir = os.path.join(model_dir, "fine_1.pth")
        trainer.best_path = save_model_dir

        # trainer.train()

        for index in range(1, 6):
            adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler = load_data( args, index)
            trainer.test(model, args, test_dataloader, scaler, trainer.logger, save_path=save_model_dir)
    elif args.mode == "fine":
        for index in range(2, 6):
            adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler = load_data( args, index)

            model, loss, optimizer, lr_scheduler = generate_model_components(args)

            # load model
            if index != 1:
                if index == 2:
                    last_model_dir = os.path.join(model_dir, f"fine_{index - 1}.pth")
                else:
                    last_model_dir = os.path.join(model_dir, f"fine_{index - 1}.pth")
                model.load_state_dict(torch.load(last_model_dir))


            trainer = Trainer(
                args=args,
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                test_loader=test_dataloader,
                scaler=scaler,
                model=model,
                loss=loss,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )

            trainer.logger.info(f"{index} ")

            save_model_dir = os.path.join(model_dir, f"fine_test_{index}.pth")
            trainer.best_path = save_model_dir

            trainer.train()
            # trainer.test(model, args, test_dataloader, scaler, trainer.logger, save_path=save_model_dir)
    elif args.mode == "mkd":
        for alpha in [0.01]:
            # for model_lambda in float_range(0.3, 0.7, 0.2):
            for model_lambda in [0.57]:
                for stu in [2]:
                    for compress in [1]:
                        # for prune in ['rand', 'mag', 'snip', 'grasp', 'synflow']:
                        #     args.pruner = prune
                        #     if prune == "synflow":
                        #         args.prune_epochs = 100
                        #     else:
                        #         args.prune_epochs = 1
                        for index in range(2, 6):
                            adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args, index)
                            model, loss, optimizer, lr_scheduler = generate_model_components(args)
                            # load model


                            # 检查可用设备
                            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                            # 根据索引加载模型
                            if index == 2:
                                last_model_dir = os.path.join(model_dir, "fine_1.pth")
                            else:
                                last_model_dir = os.path.join(model_dir,
                                                              f"mkd_{index - 1}_{alpha}_{model_lambda}_{compress}_{args.pruner}_{stu}.pth")

                            # 加载模型状态字典并映射到正确设备
                            model.load_state_dict(torch.load(last_model_dir, map_location=device))

                            teacher_model = copy.deepcopy(model).to(args.device)

                            trainer = Trainer(
                                args=args,
                                train_loader=train_dataloader,
                                val_loader=val_dataloader,
                                test_loader=test_dataloader,
                                scaler=scaler,
                                model=model,
                                loss=loss,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                freeze=True
                            )

                            trainer.logger.info("w/o  time_noise ")

                            trainer.logger.info(f"{index} {alpha} {model_lambda} {compress} {stu} ")
                            prune_model(args, model, trainer, compress)

                            save_model_dir = os.path.join(model_dir, f"mkd_{index}_{alpha}_{model_lambda}_{compress}_{args.pruner}_{stu}.pth")
                            trainer.best_path = save_model_dir

                            trainer.mkd_train(teacher_model, alpha, model_lambda, adj_mx, stu)














