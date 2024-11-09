import os
import torch
import argparse
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='svhn, cifar10, cinic10')  
parser.add_argument('--method', type=str, default='fedavg')  # algorithm
parser.add_argument('--fluctuate', type=int, default=1, help='1: 0.4, 2: 0.2, 3: 0.1')
parser.add_argument('--lr', type=float, default=0.05)  
parser.add_argument('--lr_global', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--total_rounds', type=int, default=200)
parser.add_argument('--seeds', type=str, default='3')  
parser.add_argument('--eval_freq', type=int, default=2)
parser.add_argument('--num_clients', type=int, default=100)
parser.add_argument('--gpu', type=int, default=1)  
parser.add_argument('--dirich_alpha', type=float, default=0.1)
parser.add_argument('--sigma', type=float, default=10.)


args = parser.parse_args()

print(', '.join(f'{k}={v}' for k, v in vars(args).items()))

algorithm  = args.method

gpu = bool(args.gpu) and torch.cuda.is_available()
device = 'cuda' if gpu else 'cpu'

dataset_dict ={
    'cifar10': 'CIFAR10',
    'svhn'   : 'SVHN',
    'cinic10': 'CINIC10'
}


if args.dataset == 'svhn':
    model_name = 'cnnsvhn'

elif args.dataset == 'cifar10':
    model_name = 'cnncifar10'

elif args.dataset == 'cinic10':
    model_name = 'cnncinic10'

else:
    raise NotImplementedError



dataset      = dataset_dict[args.dataset]


total_rounds = args.total_rounds 


seed_str     = args.seeds.split(',')
seeds        = [int(i) for i in seed_str]

dataset_file_path = os.path.join(os.path.dirname(__file__), 'raw_data')

dirichlet_alpha   = args.dirich_alpha

fluctuate         = args.fluctuate
sigma             = args.sigma

num_clients       = args.num_clients
lr_local_init     = args.lr
lr_global         = args.lr_global


batch_size_train  = args.batch_size
eval_freq         = args.eval_freq


prefix = 'results' + '_' + dataset + '_' + model_name + '_' + algorithm + '_lr' + str(lr_local_init) + \
                        '_num_clients' + str(num_clients) +\
                        '_lr_global' + str(lr_global) + '_dataAlpha' + str(dirichlet_alpha) + \
                        '_fluctuate' + str(fluctuate) + '_sigma'     + str(sigma)     


if dataset == 'CIFAR10' or dataset == 'CINIC10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    transform_train_eval = None
    
elif dataset == 'SVHN':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
    ])
    transform_train_eval = None

else:
    pass