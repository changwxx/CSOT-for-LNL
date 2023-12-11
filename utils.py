import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import normalize
import math
import glob, os, shutil
from torch import nn
from plabel_allocator import *

class NegEntropy(object):
    def __call__(self, outputs):
        probs = F.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))
    
def test_solo(epoch, net, test_log, test_loader, args):
    net.eval()
    
    correct = 0
    total = 0
    num = len(test_loader.dataset)
    all_outputs = torch.zeros((num, args.num_class)).cuda()
    gt_labels = torch.zeros((num), dtype=torch.int64).cuda()
    with torch.no_grad():
        for _, (inputs, targets, idx) in enumerate(test_loader):
            
            inputs, targets = inputs.cuda(), targets.cuda()
            feat, outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            
            idx = idx.cuda()
            all_outputs[idx,:] = outputs
            gt_labels[idx] = targets

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    return acc

def mixup(inputs, targets, alpha):
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)
    idx = torch.randperm(inputs.size(0))
    input_a, input_b = inputs, inputs[idx]
    target_a, target_b = targets, targets[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b
    return mixed_input, mixed_target

def warmup(epoch, net, optimizer, dataloader, CEloss, args, conf_penalty, log, net_arg):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, gt_labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        _, out = net(inputs)
        loss = CEloss(out, labels)

        if args.noise_mode == 'asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(out)
            L = loss + penalty
        elif args.noise_mode == 'sym':
            L = loss
        L.backward()
        optimizer.step()
        
        net_arg["iter"] += 1

def curriculum_scheduler(t, T, begin=0, end=1, mode=None, func=None):
    """
    ratio \in [0,1]
    """
    pho = t/T
    if mode == 'linear':
        ratio = pho
    elif mode == 'exp':
        # ratio = 1 - math.exp(-5*pho)
        ratio = 1 - math.exp(-4*pho)
    elif mode == 'customize':
        ratio = func(t, T)
    budget = begin + ratio * (end-begin)
    return budget, pho

def train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader, 
            u_argmax_plabels, u_conf_u_mask, args, log, net_arg, semi_flag=True):
    net.train()

    if labeled_trainloader is not None:
        lab_train_iter = iter(labeled_trainloader)
        num_iter_lab = len(labeled_trainloader)
    else:
        num_iter_lab = 0

    if unlabeled_trainloader is not None:
        unlab_train_iter = iter(unlabeled_trainloader)
        num_iter_unlab = len(unlabeled_trainloader)
    else:
        num_iter_unlab = 0
    
    num_iter = max(num_iter_lab, num_iter_unlab)

    temp = 0.5
    def mixmatch_loss(inputs, targets, net, args):
        mixed_inputs = torch.cat(inputs, dim=0)
        mixed_targets = torch.cat(targets, dim=0)
        mixed_input, mixed_target = mixup(mixed_inputs, mixed_targets, alpha=args.alpha)
        _, mixed_logits = net(mixed_input)

        # mixup regularization for labeled data
        loss = -torch.mean(torch.sum(F.log_softmax(mixed_logits, dim=1) * mixed_target, dim=1))

        # penalty regularization for mixed labeled data
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(mixed_logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))
        return loss, penalty

    for batch_idx in range(num_iter):
        
        if num_iter_lab > 0:
            try:
                inputs_xw, inputs_xw2, inputs_xs, labels_x, gt_labels_x, conf_x, index_x = lab_train_iter.next()
            except:
                lab_train_iter = iter(labeled_trainloader)
                inputs_xw, inputs_xw2, inputs_xs, labels_x, gt_labels_x, conf_x, index_x = lab_train_iter.next()
            
            inputs_x = [inputs_xw.cuda(), inputs_xw2.cuda()]

            _, _, out1, out2, _, _ = net(inputs_x, ssl=True)

            target_x = torch.eye(args.num_class, dtype=torch.float64)[labels_x].cuda().detach()
            
            # label refinement (refer to DivideMix)
            conf_x = conf_x.view(-1, 1).type(torch.FloatTensor).cuda()
            # conf_x = 1
            with torch.no_grad(): 
                px = (torch.softmax(out1, dim=1) + torch.softmax(out2, dim=1)) / 2
                px = conf_x * target_x + (1 - conf_x) * px
                ptx = px ** (1 / temp)  # temparature sharpening
                target_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                target_x = target_x.detach()
            
            Lx, penalty = mixmatch_loss(inputs_x, [target_x, target_x], net, args)

            _, out_xs = net(inputs_xs.cuda())
            px_w = torch.softmax(out1, dim=1)
            target_x = conf_x * target_x + (1 - conf_x) * px_w
            L_fix_x = -torch.mean(torch.sum(F.log_softmax(out_xs, dim=1) * target_x, dim=1))

        else:
            Lx = torch.tensor([0]).cuda()
            penalty = torch.tensor([0]).cuda()
            L_fix_x = torch.tensor([0]).cuda()

        if num_iter_unlab > 0:
            try:
                inputs_uw, inputs_uw2, inputs_us, labels_u, gt_labels_u, conf_u, index_u = unlab_train_iter.next()
            except:
                unlab_train_iter = iter(unlabeled_trainloader)
                inputs_uw, inputs_uw2, inputs_us, labels_u, gt_labels_u, conf_u, index_u = unlab_train_iter.next()

            
            inputs_u = [inputs_uw.cuda(), inputs_us.cuda()]
            z1, z2, out_uw, out_us, p1, p2 = net(inputs_u, ssl=True)

            criterion = torch.nn.CosineSimilarity(dim=1).cuda()
            loss_rep_unlab = -(criterion(p1, z2.detach()).mean() + criterion(p2, z1.detach()).mean()) * 0.5

            if semi_flag:
                index_u = index_u.cuda()
                u_plabels = u_argmax_plabels[index_u]
                u_conf_mask = u_conf_u_mask[index_u]

                pu_w = torch.softmax(out_uw, dim=1)
                id_row = torch.ones((pu_w.size(0),), dtype=torch.bool).cuda()
                max_probs = pu_w[id_row, u_plabels]
                p_cutoff = 0.95
                mask = max_probs.ge(p_cutoff)
                mask = torch.logical_and(mask, u_conf_mask)
                targets_u = torch.eye(args.num_class, dtype=torch.float64).cuda()[u_plabels]
                L_fix_u = -torch.mean(mask.float()*torch.sum(F.log_softmax(out_us, dim=1) * targets_u, dim=1))
                
            else:
                L_fix_u = torch.tensor([0]).cuda()
        else:
            L_fix_u = torch.tensor([0]).cuda()
            loss_rep_unlab = torch.tensor([0]).cuda()

        net_arg["iter"] += 1
        
        # overall loss
        loss_all = Lx + penalty + L_fix_x + L_fix_u + loss_rep_unlab

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()


def output_selected_rate(conf_l_mask, conf_u_mask, lowconf_u_mask, log, global_iter, net_id):
    selected_rate_conf_l = torch.sum(conf_l_mask)/conf_l_mask.size(0)
    selected_rate_conf_u = torch.sum(conf_u_mask)/conf_u_mask.size(0)
    selected_rate_lowconf_u = torch.sum(lowconf_u_mask)/lowconf_u_mask.size(0)
    return selected_rate_conf_l, selected_rate_conf_u, selected_rate_lowconf_u

def get_masks(argmax_plabels, noisy_labels, gt_labels, selected_mask):
    with torch.no_grad():
        equal_label_mask = torch.eq(noisy_labels, argmax_plabels)
        conf_l_mask = torch.logical_and(selected_mask, equal_label_mask)
        conf_u_mask = torch.logical_and(selected_mask, ~equal_label_mask)
        lowconf_u_mask = ~selected_mask
        return conf_l_mask, conf_u_mask, lowconf_u_mask


def CSOT_PL(net, eval_loader, num_class, batch_size, feat_dim=512, budget=1., sup_label=None, 
              reg_feat=0.5, reg_lab=0.5, version='fast', Pmode='out', reg_e=0.01, reg_sparsity=None):
    net.eval()

    all_pseudo_labels = torch.zeros((len(eval_loader.dataset), num_class), dtype=torch.float64).cuda()
    all_noisy_labels = torch.zeros((len(eval_loader.dataset),), dtype=torch.int64).cuda()
    all_gt_labels = torch.zeros((len(eval_loader.dataset),), dtype=torch.int64).cuda()
    all_selected_mask = torch.zeros((len(eval_loader.dataset)), dtype=torch.bool).cuda()
    all_conf = torch.zeros((len(eval_loader.dataset),), dtype=torch.float64).cuda()
    all_argmax_plabels = torch.zeros((len(eval_loader.dataset),), dtype=torch.int64).cuda()
    # loading given samples
    for batch_idx, (inputs, labels, gt_labels, index) in enumerate(eval_loader):
        feat, out = net(inputs.cuda())
        index = index.cuda()

        if sup_label is not None:
            L = torch.eye(num_class, dtype=torch.float64)[sup_label[index]].cuda()
        else:
            L = torch.eye(num_class, dtype=torch.float64)[labels].cuda()

        if Pmode == 'out':
            P = out
        if Pmode == 'logP':
            P = F.log_softmax(out, dim=1)
        if Pmode == 'softmax':
            P = F.softmax(out, dim=1)
        
        norm_feat = F.normalize(feat)
        couplings, selected_mask = curriculum_structure_aware_PL(norm_feat.detach(), P.detach(), top_percent=budget, L=L, 
                                                                reg_feat=reg_feat, reg_lab=reg_lab, version=version, reg_e=reg_e,
                                                                reg_sparsity=reg_sparsity)
        all_noisy_labels[index] = labels.cuda()
        all_gt_labels[index] = gt_labels.cuda()
        all_selected_mask[index] = selected_mask

        row_sum = torch.sum(couplings, 1).reshape((-1,1))
        pseudo_labels = torch.div(couplings, row_sum)
        max_value, argmax_plabels = torch.max(couplings, axis=1)
        conf = max_value / (1/couplings.size(0))
        conf = torch.clip(conf, min=0, max=1.0)

        all_conf[index] = conf
        all_pseudo_labels[index, :] = pseudo_labels
        all_argmax_plabels[index] = argmax_plabels

    return all_pseudo_labels, all_noisy_labels, all_gt_labels, all_selected_mask, all_conf, all_argmax_plabels