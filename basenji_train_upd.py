import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
import torch.nn.functional as F

# import ray
from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

import json
from itertools import groupby



import pyBigWig

from basenji_modules_memmap import * 
from basenji_model import *
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-batch_size', '--batch_size', type=int, default = 1, help='Batch size')
    parser.add_argument('-num_epochs', '--num_epochs' , default = 10, type=int, help='Number of epochs')
    parser.add_argument('-seq_len', '--seq_len' , default = 131072, type=int, help='Sequence length')
    parser.add_argument('-chroms', '--chroms' , default = 'all', type=str, help='Chromosomes to train on')

    
    parser.add_argument('-lr', '--lr' , default = 0.001, type=float, help='Learning rate')
    
    parser.add_argument('-input_file', '--input_file',  type=str, help='Path to the input')
    parser.add_argument('-target_file', '--target_file', type=str, help='Path to the targets')
    parser.add_argument('-debug', '--debug', type=bool, default=False,  help='Path to the targets')

    return parser.parse_args()


def main():
    args = get_args()
    print ('Got the args')
    memmap_data_contigs_dir = os.path.join(os.getcwd(), 'memmaps')
    target = pyBigWig.open('CNhs12843.bw', 'r')
    target.chroms()
    chrom_seq = target.chroms()

    model = BasenjiModel(debug=False, seq_len=args.seq_len*args.batch_size, loss='poisson', lr=0.1, opt='SGD', momentum=0.99)
    model.compile(device='cuda:3')    
    print ('Compiled the model')
    print ("Model seq len", model.seq_len)
    start = time()
    res = train_model(model, epochs=args.num_epochs, debug=False)
    print ('TOTAL TIME', time() - start)
    np.save("/wynton/home/goodarzi/ddyachkova/basenji_pytorch/training_30_epochs_chr1_memmap.npy", res)
    
if __name__ == '__main__':
     main()
