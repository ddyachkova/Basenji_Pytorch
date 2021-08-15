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
import itertools
from itertools import groupby
import gzip 
from io import BytesIO
from time import time 
import os 
#import matplotlib.pyplot as plt

import pyBigWig

class DNA_Iter(Dataset):
    def __init__(self, input_name, target_name, batch_size, chrom, target_window = 128):
        self.batch_size = batch_size
        self.chrom = chrom

        self.seq = self.read_memmap_input(input_name)
        self.target_name = target_name
        self.target = target_name.values(chrom, 0, target_name.chroms()[chrom], numpy=True)
        self.target_window = target_window
        self.nucs = np.array(["A", "T", "C", "G"])
    
    def __len__(self):
      return int(len(self.seq) / self.batch_size)

    def __getitem__(self, idx):      
      seq_subset = self.seq[idx:idx + self.batch_size]
      dta = self.dna_1hot(seq_subset)

      tgt = self.target[idx:idx+self.batch_size]
      tgt_window = self.calc_mean_lst(tgt, self.target_window)
      return torch.tensor(dta), torch.tensor(tgt_window) #.view(4, self.batch_size), torch.tensor(tgt_window)

    def read_bw_target(self, bw_name):
        target = pyBigWig.open(bw_name, 'r')
        return target

    def read_numpy_input(self, np_gq_name):
      # np_seq = gzip.open(np_gq_name, "rb").read()
      # seq = np.load(BytesIO(np_seq))
      seq = np.load(np_gq_name)
      return seq

    def read_memmap_input(self, mmap_name):
      seq = np.memmap(mmap_name, dtype='S8') #, mode='r')
      return seq


    def dna_1hot(self, seq):
      adtensor = np.zeros((4, self.batch_size), dtype=float)
      for nucidx in range(len(self.nucs)):
          nuc = self.nucs[nucidx].encode()
          j = np.where(seq[0:len(seq)] == nuc)[0]
          adtensor[nucidx, j] = 1
      return adtensor


    def calc_mean_lst(self, lst, n):
        return np.array([np.mean(lst[i:i + n]) for i in range(int(len(lst)/n))])
      
def get_train_val_loader(input_files_dir, y, batch_size):
    target_bw = pyBigWig.open(y, 'r')
    np.random.seed(42)
    files_idx = np.arange(len(os.listdir(input_files_dir)))
    np.random.shuffle(files_idx)
    input_files = os.listdir(input_files_dir)
    valid_dset = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, input_files[i]), target_bw, 128, ('chr' + str(i+1)))for i in files_idx[:1]])
    training_dset = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, input_files[i]), target_bw, 128, ('chr' + str(i+1)))for i in files_idx[2:12]])
    train_loader = DataLoader(dataset=training_dset, batch_size=batch_size, shuffle=True, sampler=None) #, num_workers=4)
    val_loader = DataLoader(dataset=valid_dset, batch_size=batch_size, shuffle=True, sampler=None) #, num_workers=4)
    return train_loader, val_loader




def _get_conv1d_out_length(l_in, padding, dilation, kernel, stride):
    return int((l_in + 2*padding - dilation*(kernel-1) - 1)/stride)+1


def pearsonr_pt(x, y):
    mean_x, mean_y= torch.mean(x), torch.mean(y)
    xm, ym = x.sub(mean_x), y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def get_input(batch): 
  seq_X,y = batch
  # try:
  #   seq_X, y = seq_X.type(torch.FloatTensor).view(1, 4, model.seq_len).cuda(), y.view(1, int(model.seq_len/128)).float().cuda()
  # except: 
  seq_X, y = seq_X.type(torch.FloatTensor).view(1, 4, seq_X.shape[0]*seq_X.shape[-1]).cuda(), y.view(1, y.shape[0]).float().cuda()
  return seq_X,y

def get_pred_loss(model, seq_X, y):
  out = model(seq_X).view(y.shape)
  loss = model.loss_fn(out,y)
  R = pearsonr_pt(out.squeeze(), y.squeeze()).to('cpu').detach().numpy()
  return loss, R


def train_model(input_file, target_file, chroms, model, epochs, clip=10, device='cpu', modelfile='models/best_ret_checkpoint.pt', logfile = None, tuneray=False, verbose=True, debug=False):
    train_losses, valid_losses, train_Rs, valid_Rs = [], [], [], []
    best_model=-1
    for epoch in range(epochs):
        # train_loader, val_loader = get_train_val_loader('chr1_sequence.numpy.gz', 'basenji_target', model.seq_len, 'chr1', cut=0.2)
        train_loader, val_loader = get_train_val_loader(input_file, target_file, int(model.seq_len/128))

        # train_loader, val_loader = get_train_val_loader('hg19.ml.fa', 'basenji_target', model.seq_len, 'chr1', cut=0.2)
        print (len(train_loader))
        # training_loss, train_R = 0.0, 0.0
        for batch_idx, batch in enumerate(train_loader):
            model.train()
            model.optimizer.zero_grad()
            seq_X, y = get_input(batch)
            if debug: 
              print ('X', seq_X.shape, 'y', y.shape)
            if seq_X.shape[-1] == int(model.seq_len): 
              
              loss, R = get_pred_loss(model, seq_X, y)
              loss.backward()
              
              torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
              #torch.nn.utils.clip_grad_value_(self.parameters(), clip)
              
              model.optimizer.step()
              train_losses.append(loss.data.item())
              train_Rs.append(R)
              
              if verbose and batch_idx%10==0:
                  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tR: {:.6f}'.format(
                          epoch, int(batch_idx * train_loader.batch_size/1024), len(train_loader), 100. * batch_idx * float(train_loader.batch_size) / len(train_loader.dataset),
                          loss.item(), R))
            else: 
              continue                
        if val_loader:
            target_list, pred_list = [], []
            valid_loss, valid_R = 0.0, 0.0
            model.eval()
            
            for batch_idx, batch in enumerate(val_loader):
                seq_X, y = get_input(batch) 
                if seq_X.shape[-1] == int(model.seq_len):
                  out = model(seq_X).view(y.shape)
                  loss, R = get_pred_loss(model, seq_X, y)
                  
                  valid_loss += loss.data.item() #* seq_X.size(0)
                  valid_R += R
                else: 
                  continue
            valid_Rs.append(valid_R / len(val_loader.dataset))
            valid_losses.append(valid_loss / len(val_loader.dataset))

            if tuneray:
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), model.optimizer.state_dict()), path)
                tune.report(loss=valid_loss, corel=valid_R)

            if verbose:
                print('Validation: Loss: {:.6f}\tR: {:.6f}'.format(valid_loss, valid_R))
            
            if logfile:
                logfile.write('Validation: Loss: {:.6f}\tR: {:.6f}\n'.format(valid_loss, valid_R))

            if (valid_R>best_model):
                best_model = valid_R
                if modelfile:
                    print('Best model updated.')
#                         self.save_model(model, modelfile)
            
            
    return {'train_loss':train_losses, 'train_Rs':train_Rs, 'valid_loss':valid_losses, 'valid_Rs':valid_Rs}

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
