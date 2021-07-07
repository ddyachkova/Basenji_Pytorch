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


class DNA_Iter(Dataset):
    def __init__(self, seq, chrom, batch_size, target):
        self.batch_size = batch_size

        self.seq = seq
        self.chrom = chrom
        self.target = target
        self.target_window = 128
    
    def __len__(self):
        return int(len(self.seq) // self.batch_size)

    def __getitem__(self, idx):
        dta = self.dna_1hot(self.seq[self.batch_size * idx:self.batch_size*(idx+1)], n_uniform=True)
#         tgt = self.target.values(self.chrom, self.batch_size_y * idx, self.batch_size_y*(idx+1))
        tgt = self.target[self.batch_size * idx: self.batch_size*(idx+1)]
        tgt_window = self.calc_mean_lst(tgt, self.target_window)
        return torch.tensor(dta), torch.tensor(tgt_window) #.view(4, self.batch_size), torch.tensor(tgt_window)

    def dna_1hot(self, seq, seq_len=None, n_uniform=False):
        if seq_len is None:
            seq_len = len(seq)
            seq_start = 0
        else:
            if seq_len <= len(seq):
              # trim the sequence
                seq_trim = (len(seq) - seq_len) // 2
                seq = seq[seq_trim:seq_trim + seq_len]
                seq_start = 0
            else:
                seq_start = (seq_len - len(seq)) // 2

        seq = seq.upper()

          # map nt's to a matrix len(seq)x4 of 0's and 1's.
        if n_uniform:
            seq_code = np.zeros((seq_len, 4), dtype='float16')
        else:
            seq_code = np.zeros((seq_len, 4), dtype='bool')

        for i in range(seq_len):
            if i >= seq_start and i - seq_start < len(seq):
                nt = seq[i - seq_start]
                if nt == 'A':
                    seq_code[i, 0] = 1
                elif nt == 'C':
                    seq_code[i, 1] = 1
                elif nt == 'G':
                    seq_code[i, 2] = 1
                elif nt == 'T':
                    seq_code[i, 3] = 1
                else:
                    continue

        return seq_code
    
    def calc_mean_lst(self, lst, n):
        return np.array([mean(lst[i*n:(i+1) *n]) for i in range(int(len(lst)/n))])


class Faiter(Dataset):
    """Main DataSet"""

    def __init__(self, fasta_name, target_name, batch_size, chroms='all'):
        
        self.fiter = self.fasta_iter(fasta_name)
#         self.target = self.read_bw_target(target_name)
        self.target = np.load(target_name)
        self.batch_size = batch_size

        self.datasets = {}
        self.len = 0
#         self.chroms = chroms
        for ff in self.fiter:
            headerStr, seq = ff
            print (headerStr)
            if chroms != 'all':
                if headerStr == chroms: 
                    dta_iter = DNA_Iter(seq, headerStr, self.batch_size, self.target)
                    self.len += len(dta_iter)
                    self.datasets[headerStr] = iter(dta_iter)
                    break 

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        class_id = index % len(self.datasets.values())
        dataset_key = list(self.datasets.keys())[class_id]
        dataset = self.datasets[dataset_key]
        item = next(dataset)
#         print (dataset_key)
        return item
    
    def read_bw_target(self, bw_name):
        target = pyBigWig.open(bw_name, 'r')
        return target
    
    def fasta_iter(self, fasta_name):
        """
        modified from Brent Pedersen
        Correct Way To Parse A Fasta File In Python
        given a fasta file. yield tuples of header, sequence
        """
        fh = open(fasta_name)
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

        for header in faiter:
            headerStr = header.__next__()[1:].strip()
            seq = "".join(s.strip() for s in faiter.__next__())

            yield (headerStr, seq)


def get_train_val_loader(X, y, seq_len, batch_size, chroms, cut=0.2):
#     dset = Faiter('data/heart_l131k/hg19.ml.fa', 'chr21_arr.npy', 16384)
    dset = Faiter(X, y, batch_size, chroms)
    print (len(dset), 'len dset')
    dset_indices = list(range(len(dset)))
    np.random.shuffle(dset_indices)
    val_split_index = int(np.floor(cut * len(dset_indices)))

    train_idx, val_idx = dset_indices[val_split_index:], dset_indices[:val_split_index]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset=dset, batch_size=1, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(dataset=dset, batch_size=1, shuffle=False, sampler=val_sampler)
    return train_loader, val_loader


