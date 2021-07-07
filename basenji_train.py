from itertools import groupby
import random
from torch.utils.data import *
# import pyBigWig
import sys 
import collections
import scipy

import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import scipy.stats as stats

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from itertools import chain
from statistics import mean
import argparse 

from basenji_modules import * 
from basenji_model import *


def get_args():
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-batch_size', '--batch_size', type=int, default = 1, help='Batch size')
    parser.add_argument('-num_epochs', '--num_epochs' , default = 10, type=int, help='Number of epochs')
    parser.add_argument('-seq_len', '--seq_len' , default = 131072, type=int, help='Sequence length')
    parser.add_argument('-chroms', '--chroms' , default = 'all', type=str, help='Chromosomes to train on')

    
    parser.add_argument('-lr', '--lr' , default = 0.001, type=float, help='Learning rate')
    
    parser.add_argument('-input', '--input',  type=str, help='Path to the input')
    parser.add_argument('-target', '--target', type=str, help='Path to the targets')
    parser.add_argument('-debug', '--debug', type=bool, default=False,  help='Path to the targets')

    return parser.parse_args()


def main():
    args = get_args()
    print ('Got the args')
    train_loader, val_loader = get_train_val_loader(args.input, args.target, args.seq_len, args.batch_size,  args.chroms, cut=0.2)
    print ('Got the loader')
    model = retDNNModel(debug=False, seq_len=args.seq_len, loss='poisson')
    model.compile(device='cuda')
    print ('Compiled the model')
    model.train_model(train_loader, val_loader, epochs=args.num_epochs, device='cuda', debug=args.debug)

if __name__ == '__main__':
     main()
