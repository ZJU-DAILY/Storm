import copy
import sys

sys.path.append('../')

import os
import torch
from datetime import datetime
from lib.utils import *
from lib.data_loader import *
from lib.generate_adj_mx import *
from lib.evaluate import MAE_torch
from STGCN_Config import args
from STGCN_Utils import *
from STGCN_Trainer import Trainer
from model.STGCN.stgcn import STGCN as Network
import numpy as np
import random


def set_env(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(args, index):
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(index, args,
                                                                               normalizer=args.normalizer,
                                                                               tod=False,
                                                                               dow=False,
                                                                               single=False)
    if args.graph_type == 'BINARY':
        adj_mx = get_adjacency_matrix(args.graph_path, args.num_node, type='connectivity', id_filename=args.filename_id)
    elif args.graph_type == 'DISTANCE':
        adj_mx = get_Gaussian_matrix(args.graph_path, args.num_node, args.normalized_k, id_filename=args.filename_id)
    print("The shape of adjacency matrix : ", adj_mx.shape)
    adj_mx = torch.FloatTensor(adj_mx).to(args.device)
    return adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler


def generate_model_components(args, adj_mx):
    adj_mx = np.array(adj_mx.cpu())
    L = scaled_laplacian(adj_mx)
    Lk = cheb_poly(L, Ks=args.KS)
    Lk = torch.Tensor(Lk.astype(np.float32)).to(args.device)
    # 1. model
    model = Network(
        ks=args.KS,
        kt=args.KT,
        bs=args.channels,
        T=args.horizon,
        n=args.num_node,
        Lk=Lk,
        p=args.dropout
    )
    model = model.to(args.device)
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

    # optimizer = SynFlowAdam(
    #     params=model.parameters(),
    #     lr=args.lr_init,                  # 使用您的初始学习率
    #     weight_factor=0.01,                # 根据需求调整
    #     weight_decay=args.weight_decay,    # 使用您的权重衰减系数
    #     eps=1.0e-8,                       # 类似于 Adam 的 epsilon
    #     device=args.device                     # 确保 device 一致
    # )

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
    dataloader = trainer.data_loader

    prune_x = []
    prune_y = []
    for i, (data, target, _) in enumerate(dataloader['train_loader'].get_iterator()):
        prune_x.append(data)
        prune_y.append(target)
        if i + 1 >= args.prune_dataset_ratio:
            break

    data_x = np.concatenate(prune_x, axis=-1)
    data_y = np.concatenate(prune_y, axis=-1)

    # 创建一个新的 DataLoader 对象，这就是你的 prune_loader
    prune_loader = DataLoaderM(data_x, data_y, data_x, batch_size=dataloader['train_loader'].batch_size)

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

        model, loss, optimizer, lr_scheduler = generate_model_components(args, adj_mx)
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
            cl=args.cl,
            new_training_method=args.new_training_method
        )

        save_model_dir = os.path.join(model_dir, "one.pth")
        trainer.best_path = save_model_dir

        trainer.train()

        for index in range(2, 6):
            adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args, index)
            trainer.test(model, args, test_dataloader, scaler, trainer.logger, save_path=save_model_dir)
    elif args.mode == "fine":
        for index in range(2, 6):
            adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args, index)

            model, loss, optimizer, lr_scheduler = generate_model_components(args, adj_mx)

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
                lr_scheduler=lr_scheduler,
                cl=args.cl,
                new_training_method=args.new_training_method,
                freeze=False
            )

            trainer.logger.info(f" {index} ")

            save_model_dir = os.path.join(model_dir, f"fine_test_{index}.pth")
            trainer.best_path = save_model_dir

            trainer.train()
            # trainer.test(model, args, test_dataloader, scaler, trainer.logger, save_path=save_model_dir)
    elif args.mode == "mkd":

        for alpha in [0.1]:
            # for model_lambda in float_range(0.3, 0.7, 0.2):
            for model_lambda in [0.55]:
            # for model_lambda in float_range(0.3, 0.7, 0.2):
                # for compress in float_range(0.3, 1, 0.7):
                for save_model in [1]:
                    for index in range(2, 6):
                        adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args, index)
                        model, loss, optimizer, lr_scheduler = generate_model_components(args, adj_mx)

                        # load model
                        if index == 2:
                            last_model_dir = os.path.join(model_dir, "fine_1.pth")
                            model.load_state_dict(torch.load(last_model_dir, map_location=torch.device('cpu')))

                        else:
                            last_model_dir = os.path.join(model_dir,
                                                          f"mkd_{index - 1}_{alpha}_{model_lambda}_{freeze}_{save_model}.pth")
                            model.load_state_dict(torch.load(last_model_dir, map_location=torch.device('cpu')))

                        teacher_model = copy.deepcopy(model).to(args.device)

                        freeze = True
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
                            cl=args.cl,
                            new_training_method=args.new_training_method,
                            freeze=freeze
                        )

                        trainer.logger.info(f"{index} {alpha} {model_lambda} {freeze}  {save_model}")

                        save_model_dir = os.path.join(model_dir,
                                                      f"mkd_{index}_{alpha}_{model_lambda}_{freeze}_{save_model}.pth")
                        trainer.best_path = save_model_dir

                        trainer.mkd_train(teacher_model, alpha, model_lambda, adj_mx, save_model)
