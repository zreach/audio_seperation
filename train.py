#!/usr/bin/env python

import argparse

import torch
import random
from data import AudioDataLoader, AudioDataset
from solver import Solver
from net import TasNet
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICE']='2'

def init_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(
    "Time-domain Audio Separation Network (TasNet) with Permutation Invariant "
    "Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
# Network architecture
parser.add_argument('--L', default=40, type=int,
                    help='Segment length (40=5ms at 8kHZ)')
parser.add_argument('--N', default=500, type=int,
                    help='The number of basis signals')
parser.add_argument('--e_type', default='conv', type=str,choices=['conv','fc'],
                    help='The structure in encoder')
parser.add_argument('--hidden_size', default=500, type=int,
                    help='Number of LSTM hidden units')
parser.add_argument('--num_layers', default=4, type=int,
                    help='Number of LSTM layers')
parser.add_argument('--bidirectional', default=1, type=int,
                    help='Whether use bidirectional LSTM')
parser.add_argument('--nspk', default=2, type=int,
                    help='Number of speaker')
# Training config
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', '-b', default=32, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='sgd', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')

parser.add_argument('--percent', default=1,type=float,
                    help='How many percent of data will be used')
parser.add_argument('--use_cuda', default=0,type=int,
                    help='use cuda or not')
def main(args):
    # Construct Solver
    # data
    init_seed(42)
    # print(args.batch_size)
    tr_dataset = AudioDataset(args.train_dir, args.batch_size,
                              sample_rate=args.sample_rate, L=args.L)
    cv_dataset = AudioDataset(args.valid_dir, args.batch_size,
                              sample_rate=args.sample_rate, L=args.L)
    sample_num = int(args.percent * len(tr_dataset))
    print(sample_num)
    tr_loader = AudioDataLoader(tr_dataset[:sample_num], batch_size=args.batch_size,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
    cv_loader = AudioDataLoader(cv_dataset[:sample_num], batch_size=args.batch_size,
                                shuffle=args.shuffle,
                                num_workers=0)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    model = TasNet(args.L, args.N, args.hidden_size, args.num_layers,
                   bidirectional=args.bidirectional, nspk=args.nspk,e_type =args.e_type)
    print(model)
    # print(args.use_cuda)
    if args.use_cuda:
        model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return
        
    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
