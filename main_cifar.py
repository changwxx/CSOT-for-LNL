from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import os, datetime, argparse
from PreResNet import *
import dataloader_cifar as dataloader
from utils import NegEntropy
from utils import *
import numpy as np
# from utils import create_folder_and_save_pyfile
import copy
import time
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym', help='sym or asym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--T', default=1, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=450, type=int)
parser.add_argument('--lr_switch_epoch', default=150, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--drop', default=0.0, type=float)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='/path/to/dataset', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--remark', default='debug', type=str)

parser.add_argument('--reg_feat', default=1., type=float)
parser.add_argument('--reg_lab', default=1., type=float)
parser.add_argument('--refresh_label', default=True, type=bool)
parser.add_argument('--curriculum_mode', default='linear', type=str)
parser.add_argument('--begin_rate', default=0.3, type=float)
parser.add_argument('--model_path', default='', type=str)
parser.add_argument('--Pmode', default='logP', type=str)
parser.add_argument('--reg_e', default=0.1, type=float)
parser.add_argument('--save', default=True, type=bool)
parser.add_argument('--hard', default=True, type=bool)
parser.add_argument('--Umode', default="fixmatch", type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

now = datetime.datetime.now().strftime('%b%d_%H-%M')
if args.remark == 'debug':
    root_folder = os.path.join("../", "outputs", "debug", args.dataset, f"{args.noise_mode}_{args.r}_{now}")
else:
    root_folder = os.path.join("./", "outputs", args.remark, args.dataset, f"{args.noise_mode}_{args.r}_{now}")


logger_tb = SummaryWriter(root_folder)
results_log = open(os.path.join(root_folder, 'results.txt'), 'a+')

if args.dataset == 'cifar10':
    warm_up = 10
elif args.dataset == 'cifar100':
    warm_up = 30


print('| Building net')
def create_model(args):
    model = Net(ResNet18, num_classes=args.num_class, drop=args.drop)
    model = model.cuda()
    return model
net = create_model(args)
net2 = create_model(args)
cudnn.benchmark = True

print('| Building optimizer')
optimizer = optim.SGD(list(net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CEloss = nn.CrossEntropyLoss()
if args.noise_mode == 'asym':
    conf_penalty = NegEntropy()
else:
    conf_penalty = None

loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size, num_workers=5, \
                root_dir=args.data_path, noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

warmup_trainloader = loader.run('warmup', args.batch_size*2)
test_loader = loader.run('test', args.batch_size)
eval_loader = loader.run('eval_train', 1024)


args.global_warmup_iter = warm_up*len(warmup_trainloader)

def init_netarg(id):
    net_arg = dict()
    net_arg["id"] = id
    net_arg["net"] = f"net{id}"
    net_arg["iter"] = 0
    return net_arg
net_arg = init_netarg(1)

best_acc = -1
best_epoch = -1

args.train_num = len(warmup_trainloader.dataset)
sup_label = None

scheduler = None

args.semi_flag = False
args.curriclum_epoch = 250

for epoch in range(args.num_epochs):
    # model training
    if epoch < warm_up:
        warmup(epoch, net, optimizer, warmup_trainloader, CEloss, args, conf_penalty, logger_tb, net_arg)
    else:
        reg_feat, reg_lab = args.reg_feat, args.reg_lab
        if epoch < args.curriclum_epoch:
            budget, pho = curriculum_scheduler(epoch-warm_up, args.curriclum_epoch-warm_up, 
                                            begin=args.begin_rate, end=1, mode=args.curriculum_mode)
        else:
            budget, pho = 1., 1.
        print(f"current budget = {budget} ({pho*100}%)")
        logger_tb.add_scalar(f'curriculum_budget', budget, epoch)
        Pmode = args.Pmode
        reg_e = args.reg_e
        with torch.no_grad():
            pseudo_labels1, noisy_labels, gt_labels, selected_mask, conf1, argmax_plabels = CSOT_PL(net, eval_loader, num_class=args.num_class, batch_size=args.batch_size, 
                                                                                                budget=budget, sup_label=sup_label, reg_feat=reg_feat, reg_lab=reg_lab,
                                                                                                Pmode=Pmode, reg_e=reg_e)
            

            conf_l_mask, conf_u_mask, lowconf_u_mask = get_masks(argmax_plabels, noisy_labels, None, selected_mask)
            

            _, selected_rate_conf_u, selected_rate_lowconf_u = output_selected_rate(conf_l_mask, conf_u_mask, lowconf_u_mask, logger_tb, epoch, net_id=1)

        if selected_rate_conf_u > selected_rate_lowconf_u:
            args.semi_flag = True

        unlabeled_mask1 = torch.logical_or(conf_u_mask, lowconf_u_mask)
        labeled_trainloader, unlabeled_trainloader = loader.run('train', args.batch_size, conf1.cpu().numpy(), conf_l_mask.cpu().numpy(), unlabeled_mask1.cpu().numpy())  # co-divide
        print('Train Net')
        train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader,
            argmax_plabels[unlabeled_mask1], conf_u_mask[unlabeled_mask1], 
            args, logger_tb, net_arg, semi_flag=args.semi_flag)
    
    if scheduler is None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    else:
        scheduler.step() 
    # model testing
    acc = test_solo(epoch, net, logger_tb, test_loader, args)
    print(f"now epoch = {epoch}, acc = {acc}")
    if(acc > best_acc):
        best_acc = acc
        best_epoch = epoch

    results_log.write('Epoch:%d   Now:%.2f   Best:%.2f(Epoch:%d)\n' % (epoch, acc, best_acc, best_epoch))
    results_log.flush()





