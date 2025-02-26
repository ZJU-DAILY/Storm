import sys
import copy
sys.path.append('../')
import copy
import os
import torch
import torch.nn as nn
from datetime import datetime
from lib.utils import *
from lib.data_loader import *
from lib.generate_adj_mx import *
from lib.evaluate import MAE_torch
from AGCRN_Config import args
from AGCRN_Trainer import Trainer
from model.AGCRN.agcrn import AGCRN as Network


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
    adj_mx = torch.FloatTensor(adj_mx).to(args.device)
    return adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler


def generate_model_components(args):
    init_seed(args.seed)
    # 1. model
    model = Network(
        num_node=args.num_node,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        embed_dim=args.embed_dim,
        cheb_k=args.cheb_k,
        horizon=args.horizon,
        num_layers=args.num_layers
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



if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'

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


            save_model_dir = os.path.join(model_dir, f"fine_{index}.pth")
            trainer.best_path = save_model_dir

            trainer.train()
            # trainer.test(model, args, test_dataloader, scaler, trainer.logger, save_path=save_model_dir)
    elif args.mode == "mkd":
        for alpha in [0.01]:
            for model_lambda in [0.58]:
                for stu in [2]:
                    if stu == 2:
                        save_model = 1
                    else:
                        save_model = 1
                # for compress in float_range(0, 1, 0.1):
                    for index in range(2, 6):
                        adj_mx, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args, index)
                        model, loss, optimizer, lr_scheduler = generate_model_components(args)

                        # load model
                        if index == 2:
                            last_model_dir = os.path.join(model_dir, "fine_1.pth")
                            model.load_state_dict(torch.load(last_model_dir))
                        else:
                            last_model_dir = os.path.join(model_dir,
                                                          f"mkd_{index - 1}_{alpha}_{model_lambda}_{stu}_{save_model}.pth")
                            model.load_state_dict(torch.load(last_model_dir))

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
                            freeze=False
                        )

                        trainer.logger.info(f"{index} {alpha} {model_lambda} {stu} {save_model}")
                        # prune_model(args, model, trainer, compress)

                        save_model_dir = os.path.join(model_dir, f"mkd_{index}_{alpha}_{model_lambda}_{stu}_{save_model}.pth")
                        trainer.best_path = save_model_dir

                        trainer.mkd_train(teacher_model, alpha, model_lambda, adj_mx, stu, save_model)
