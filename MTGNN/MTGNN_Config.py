import argparse
import configparser

MODE = 'mkd'

DEBUG = False
DEVICE = 'cuda:1'
MODEL = 'MTGNN'


# DATASET = 'PEMSD8'
# GRAPH = "../data/PEMSD8/PEMSD8.csv"
# FILENAME_ID = None

DATASET = 'PEMSD4'
GRAPH = "../data/PEMSD4/PEMSD4.csv"
FILENAME_ID = None

# 1. get configuration
config_file = './{}_{}.conf'.format(DATASET, MODEL)
config = configparser.ConfigParser()
config.read(config_file)


# 2. arguments parser
args = argparse.ArgumentParser(description='Arguments')
args.add_argument('--mode', default=MODE, type=str)
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--device', default=DEVICE, type=str)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--graph_path', default=GRAPH, type=str)
args.add_argument('--graph_type', default='DISTANCE', type=str)
args.add_argument('--normalized_k', default=0.1, type=float)
args.add_argument('--filename_id', default=FILENAME_ID, type=str)

# 3. conf params
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--window', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_node', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)

args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval)
# 4. model params
args.add_argument('--gcn_true', default=config['model']['gcn_true'], type=eval)
args.add_argument('--buildA_true', default=config['model']['buildA_true'], type=eval)
args.add_argument('--gcn_depth', default=config['model']['gcn_depth'], type=eval)
args.add_argument('--subgraph_size', default=config['model']['subgraph_size'], type=eval)
args.add_argument('--dropout', default=config['model']['dropout'], type=eval)
args.add_argument('--node_dim', default=config['model']['node_dim'], type=eval)
args.add_argument('--dilation_exponential', default=config['model']['dilation_exponential'], type=eval)
args.add_argument('--conv_channels', default=config['model']['conv_channels'], type=eval)
args.add_argument('--residual_channels', default=config['model']['residual_channels'], type=eval)
args.add_argument('--skip_channels', default=config['model']['skip_channels'], type=eval)
args.add_argument('--end_channels', default=config['model']['end_channels'], type=eval)
args.add_argument('--input_dim', default=config['model']['input_dim'], type=eval)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=eval)
args.add_argument('--layers', default=config['model']['layers'], type=eval)
args.add_argument('--propalpha', default=config['model']['propalpha'], type=eval)
args.add_argument('--tanhalpha', default=config['model']['tanhalpha'], type=eval)
args.add_argument('--layer_norm_affline', default=config['model']['layer_norm_affline'], type=eval)

args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
args.add_argument('--log_dir', default='./', type=str)

# 5. prune
args.add_argument('--pruner', type=str, default='rand',
                  choices=['rand', 'mag', 'snip', 'grasp', 'synflow'],
                  help='prune strategy (default: rand)')
args.add_argument('--compression', type=float, default=0.3,
                  help='quotient of prunable non-zero prunable parameters before and after pruning (default: 1.0)')
args.add_argument('--prune_epochs', type=int, default=1,
                  help='number of iterations for scoring (default: 1)')
args.add_argument('--prune_bias', type=bool, default=False,
                  help='whether to prune bias parameters (default: False)')
args.add_argument('--prune_batchnorm', type=bool, default=False,
                  help='whether to prune batchnorm layers (default: False)')
args.add_argument('--prune_layernorm', type=bool, default=False,
                  help='whether to prune batchnorm layers (default: False)')
args.add_argument('--prune_residual', type=bool, default=False,
                  help='whether to prune residual connections (default: False)')
args.add_argument('--prune_train_mode', type=bool, default=False,
                  help='whether to prune in train mode (default: False)')
args.add_argument('--reinitialize', type=bool, default=False,
                  help='whether to reinitialize weight parameters after pruning (default: False)')
args.add_argument('--shuffle', type=bool, default=False,
                  help='whether to shuffle masks after pruning (default: False)')
args.add_argument('--invert', type=bool, default=False,
                  help='whether to invert scores during pruning (default: False)')
args.add_argument('--prune_dataset_ratio', type=int, default=1,
                  help='ratio of prune dataset size ')
args.add_argument('--compression_schedule', type=str, default='exponential', choices=['linear', 'exponential'],
                  help='whether to use a linear or exponential compression schedule (default: exponential)')
args.add_argument('--mask_scope', type=str, default='global', choices=['global', 'local'],
                  help='masking scope (global or layer) (default: global)')

args = args.parse_args()
