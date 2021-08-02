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


import matplotlib.pyplot as plt

import pyBigWig


def _get_conv1d_out_length(l_in, padding, dilation, kernel, stride):
    return int((l_in + 2*padding - dilation*(kernel-1) - 1)/stride)+1


def pearsonr_pt(x, y):
    mean_x, mean_y= torch.mean(x), torch.mean(y)
    xm, ym = x.sub(mean_x), y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


class BasenjiModel(nn.Module):
    def __init__(self, n_channel=4, max_len=128, 
                 conv1kc=64, conv1ks=16, conv1st=1, conv1pd=11, pool1ks=8, pool1st=1 , pdrop1=0.2, #conv_block_1 parameters
                 conv2kc=64, conv2ks=16, conv2st=1, conv2pd=9, pool2ks=4 , pool2st=1, pdrop2=0.2, #conv_block_2 parameters
                 convdc = 6, convdkc=32 , convdks=5, #dilation block parameters
                 fchidden = 10, pdropfc=0.2, final=1, #fully connected parameters
                 
                 seq_len=131072, opt="Adam", loss="mse", lr=1e-3, momentum=0.9, weight_decay=1e-3, debug=False
                ):
        super(BasenjiModel, self).__init__()
        
        self.convdc = convdc
        self.opt = opt
        self.loss = loss
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = lr
        self.debug = debug
        self.seq_len = seq_len
        ## CNN + dilated CNN
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(n_channel, conv1kc, kernel_size=conv1ks, stride=conv1st, padding=conv1pd),
            nn.LeakyReLU(),
            nn.BatchNorm1d(conv1kc),
            nn.MaxPool1d(kernel_size=pool1ks),
            nn.Dropout(p=pdrop1),
        )
        
        conv_block_1_out_len = _get_conv1d_out_length(max_len, conv1pd, 1, conv1ks, conv1st) #l_in, padding, dilation, kernel, stride
        mpool_block_1_out_len = _get_conv1d_out_length(conv_block_1_out_len, 0, 1, pool1ks, pool1st)
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(conv1kc, conv2kc, kernel_size=conv2ks, stride=conv2st, padding=conv2pd),
            nn.LeakyReLU(),
            nn.BatchNorm1d(conv2kc),
            nn.MaxPool1d(kernel_size=pool2ks),
            nn.Dropout(p=pdrop2),
        )
        conv_block_2_out_len = _get_conv1d_out_length(mpool_block_1_out_len, conv2pd, 1, conv2ks, conv2st)
        mpool_block_2_out_len = _get_conv1d_out_length(conv_block_2_out_len, 0, 1, pool2ks, pool2st)
        
        self.dilations = nn.ModuleList()
        for i in range(convdc):
            padding = 2**(i+1)
            self.dilations.append(nn.Sequential(
                nn.Conv1d(conv2kc, convdkc, kernel_size=convdks, padding=padding, dilation=2**i),
                nn.LeakyReLU(),
                nn.BatchNorm1d(convdkc)))
        dilation_blocks_lens = []
        for i in range(convdc):
            dilation_blocks_lens.append( _get_conv1d_out_length(mpool_block_2_out_len, 2**(i+1), 2**i, convdks, 1)  * convdkc) #(l_in, padding, dilation, kernel, stride):
        
        total_length = mpool_block_2_out_len * conv2kc + sum(dilation_blocks_lens)
        self.fc = nn.Sequential(
            nn.Dropout(p=pdropfc),
            nn.Linear(256, fchidden),
            nn.ReLU(),
            nn.Dropout(p=pdropfc),
            nn.Linear(fchidden, final))
        
    def forward(self, seq):
        if self.debug: 
            print (seq.shape)
        seq = self.conv_block_1(seq)
        if self.debug: 
            print ('conv1', seq.shape)
        seq = self.conv_block_2(seq)
        if self.debug: 
            print ('conv2', seq.shape)
        seq = self.conv_block_2(seq)
        if self.debug: 
            print ('conv2', seq.shape)

        y = [] 
        for i in range(self.convdc):
            y.append(torch.flatten(self.dilations[i](seq), 1))
            if self.debug: 
                print ('dil', i, self.dilations[i](seq).shape)
        if self.debug: 
            print ('fl', torch.flatten(seq, 1).shape, 'y', y[0].shape)
        res = torch.cat([torch.flatten(seq, 1)]+y, dim=1)

        if self.debug: 
            print ('cat', res.shape)
        res = res.view(1, self.seq_len // 128, 256)
        res = self.fc(res)
        return res
        
    def compile(self, device='cpu'):
        self.to(device)
        if self.opt=="Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.opt=="SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum = self.momentum)
        if self.opt=="Adagrad":
            self.optimizer = optim.Adagrad(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        if self.loss=="mse":
            self.loss_fn = F.mse_loss
        if self.loss=="poisson":
            self.loss_fn = torch.nn.PoissonNLLLoss()

