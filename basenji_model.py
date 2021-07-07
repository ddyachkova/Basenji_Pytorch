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



def _get_conv1d_out_length(l_in, padding, dilation, kernel, stride):
    return int((l_in + 2*padding - dilation*(kernel-1) - 1)/stride)+1


def pearsonr_pt(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    

    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
#     print(r_val)
    return r_val


class retDNNModel(nn.Module):
    def __init__(self, n_channel=4, max_len=128, 
                 conv1kc=64, conv1ks=16, conv1st=1, conv1pd=11, pool1ks=8, pool1st=1 , pdrop1=0.2, #conv_block_1 parameters
                 conv2kc=64, conv2ks=16, conv2st=1, conv2pd=9, pool2ks=4 , pool2st=1, pdrop2=0.2, #conv_block_2 parameters
                 convdc = 6, convdkc=32 , convdks=5, #dilation block parameters
                 fchidden = 10, pdropfc=0.5, final=1, #fully connected parameters
                 
                 seq_len=131072, opt="Adam", loss="mse", lr=1e-3, momentum=0.9, weight_decay=1e-3, debug=False
                ):
        super(retDNNModel, self).__init__()
        
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
            nn.MaxPool1d(kernel_size=pool1ks),# stride=pool1st),
            nn.Dropout(p=pdrop1),
        )
        conv_block_1_out_len = _get_conv1d_out_length(max_len, conv1pd, 1, conv1ks, conv1st) #l_in, padding, dilation, kernel, stride
        mpool_block_1_out_len = _get_conv1d_out_length(conv_block_1_out_len, 0, 1, pool1ks, pool1st)
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(conv1kc, conv2kc, kernel_size=conv2ks, stride=conv2st, padding=conv2pd),
            nn.LeakyReLU(),
            nn.BatchNorm1d(conv2kc),
            nn.MaxPool1d(kernel_size=pool2ks),# stride=pool2st),
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
                nn.BatchNorm1d(convdkc))
            )
        dilation_blocks_lens = []
        for i in range(convdc):
            dilation_blocks_lens.append( _get_conv1d_out_length(mpool_block_2_out_len, 2**(i+1), 2**i, convdks, 1)  * convdkc) #(l_in, padding, dilation, kernel, stride):
        
        total_length = mpool_block_2_out_len * conv2kc + sum(dilation_blocks_lens)
        #print(total_length, mpool_block_2_out_len, conv2kc, dilation_blocks_lens, convdks)
        print ('total_len', total_length)
        self.fc = nn.Sequential(
            nn.Dropout(p=pdropfc),
            nn.Linear(256, fchidden),
            nn.ReLU(),
            nn.Dropout(p=pdropfc),
            nn.Linear(fchidden, final)
        )
        
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

#             y.append(self.dilations[i](seq))
        if self.debug: 
            print ('fl', torch.flatten(seq, 1).shape, 'y', y[0].shape)
        res = torch.cat([torch.flatten(seq, 1)]+y, dim=1)

        if self.debug: 
            print ('cat', res.shape)
        res = res.view(1, (self.seq_len // 128), (self.seq_len // 64))
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

            
    def train_model(self, train_loader, val_loader=None, epochs=100, clip=10, device='cpu', modelfile='models/best_ret_checkpoint.pt', logfile = None, tuneray=False, verbose=True, debug=False):
        train_losses = []
        train_Rs = []
        valid_losses = []
        valid_Rs = []
        best_model=-1
        for epoch in range(epochs):
            training_loss = 0.0
            train_R = 0.0
            if epoch == 10: 
                break
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx == 100: 
                    break
                self.train()
                self.optimizer.zero_grad()
                seq_X,y = batch
                if debug:
                    print(seq_X.shape, y.shape)

                seq_X, y = seq_X.type(torch.FloatTensor).cuda(), y.float().cuda()
                seq_X = seq_X.view(1, 4, self.seq_len)
                if debug: 
                    print (seq_X.shape)
#                 seq_X,y = seq_X.float().cuda(),y.float().cuda()
                out = self(seq_X)
                out =  out.view(1, 128)

                loss = self.loss_fn(out,y)
                if debug:
                    print(out.shape, y.shape)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
                #torch.nn.utils.clip_grad_value_(self.parameters(), clip)
                self.optimizer.step()
                training_loss = loss.data.item() * seq_X.size(0) #divide by two since every sequence is paired F/R
#                 train_R = pearsonr_pt(out[:,0],y[:,0]).to('cpu').detach().numpy()
                train_R = pearsonr_pt(out.reshape(128),y.reshape(128)).to('cpu').detach().numpy()

                train_losses.append(training_loss)
                train_Rs.append(train_R)
                
                if verbose and batch_idx%10==0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tR: {:.6f}'.format(
                            epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset), 100. * batch_idx * float(train_loader.batch_size) / len(train_loader.dataset),
                            loss.item(), train_R))
                    
            if val_loader:
                target_list = []
                pred_list = []
                valid_loss = 0.0
                valid_R = 0.0
                self.eval()
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx == 10: 
                        break

                    seq_X,y = batch
                    seq_X = seq_X.type(torch.FloatTensor)
                    seq_X = seq_X.view(1, 4, self.seq_len)

                    seq_X,y = seq_X.cuda(),y.cuda()
                    seq_X = seq_X.float().to(device)
                    y = y.float().to(device)
                    out = self(seq_X)
                    out =  out.view(1, 128)
                    loss = self.loss_fn(out[:,0],y[:,0])
                    valid_loss += loss.data.item() * seq_X.size(0)
                    pred_list.append(out.to('cpu').detach().numpy())
                    target_list.append(y.to('cpu').detach().numpy())
                targets = np.concatenate(target_list)
                preds = np.concatenate(pred_list)
                valid_R = stats.pearsonr(targets[:,0], preds[:,0])[0]
                valid_loss /= len(val_loader.dataset)

                valid_losses.append(valid_loss)
                valid_Rs.append(valid_R)
                
                if tuneray:
                    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save(
                            (self.state_dict(), self.optimizer.state_dict()), path)

                    tune.report(loss=valid_loss, corel=valid_R)

                if verbose:
                    print('Validation: Loss: {:.6f}\tR: {:.6f}'.format(valid_loss, valid_R))
                if logfile:
                    logfile.write('Validation: Loss: {:.6f}\tR: {:.6f}\n'.format(valid_loss, valid_R))

                if (valid_R>best_model):
                    best_model = valid_R
                    if modelfile:
                        print('Best model updated.')
#                         self.save_model(modelfile)
                
                
        return {'train_loss':train_losses, 'train_Rs':train_Rs, 'valid_loss':valid_losses, 'valid_Rs':valid_Rs}
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
