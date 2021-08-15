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


#import matplotlib.pyplot as plt

import pyBigWig


class DNA_Iter(Dataset):
    def __init__(self, input_name, target_name, batch_size, chrom, target_window = 128):
        self.batch_size = batch_size
        self.chrom = chrom

        self.seq = self.read_numpy_input(input_name)
        target_bw = self.read_bw_target(target_name)
        self.target = target_bw.values(chrom, 0, target_bw.chroms()[chrom], numpy=True)
        self.target_window = target_window

        self.nucs = np.array(["A", "T", "C", "G"])
        
        np.random.seed(42)
        self.indices = np.arange(target_bw.chroms()[self.chrom])
        np.random.shuffle(self.indices)
    
    def __len__(self):
      # print ('len inner dset', int(len(self.indices) / self.batch_size) )
      return int(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        ## pull a random index here with rand_ind[idx]

      indices_subset = np.array([self.indices[idx:idx + self.batch_size]][0]).astype(int)
      seq_subset = np.array([self.seq[i] for i in indices_subset])
      dta = self.dna_1hot(seq_subset)
      # tgt = self.target[idx:idx+self.batch_size]
      tgt = self.target[indices_subset]

      tgt_window = self.calc_mean_lst(tgt, self.target_window)
      return torch.tensor(dta), torch.tensor(tgt_window) #.view(4, self.batch_size), torch.tensor(tgt_window)

    def read_bw_target(self, bw_name):
        target = pyBigWig.open(bw_name, 'r')
        return target

    def read_numpy_input(self, np_gq_name):
      np_seq = gzip.open(np_gq_name, "rb").read()
      seq = np.load(BytesIO(np_seq))
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


def get_train_val_loader(X, y, batch_size, chroms, cut=0.2):
    dset = DNA_Iter(X, y, batch_size, chroms)
    dset_indices = np.arange(len(dset))
    # print ('len dset', len(dset))
    val_split_index = int(np.floor(cut * len(dset_indices)))
    train_idx, val_idx = dset_indices[val_split_index:], dset_indices[:val_split_index]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset=dset, batch_size=1, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(dataset=dset, batch_size=1, shuffle=False, sampler=val_sampler)
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

def get_input(seq_len, batch): 
  seq_X,y = batch
  seq_X, y = seq_X.type(torch.FloatTensor).view(1, 4, seq_len).cuda(), y.float().cuda()
  return seq_X,y

def get_pred_loss(model, seq_X, y):
  out = model(seq_X).view(y.shape)
  loss = model.loss_fn(out,y)
  R = pearsonr_pt(out.squeeze(), y.squeeze()).to('cpu').detach().numpy()
  return loss, R
  
def train_model(input_file, target_file, model, chroms, epochs=100, clip=10, device='cpu', modelfile='models/best_ret_checkpoint.pt', logfile = None, tuneray=False, verbose=True, debug=False):
    train_losses, valid_losses, train_Rs, valid_Rs = [], [], [], []
    best_model=-1
    for epoch in range(epochs):
        train_loader, val_loader = get_train_val_loader(input_file, target_file, model.seq_len, chroms=chroms, cut=0.2)
        # train_loader, val_loader = get_train_val_loader('hg19.ml.fa', 'basenji_target', model.seq_len, 'chr1', cut=0.2)
        print (len(train_loader))
        # training_loss, train_R = 0.0, 0.0
        for batch_idx, batch in enumerate(train_loader):
            model.train()
            model.optimizer.zero_grad()
            seq_X, y = get_input(model.seq_len, batch) 
            if debug: 
                print ('X', seq_X.shape, 'y', y.shape)
            
            loss, R = get_pred_loss(model, seq_X, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            #torch.nn.utils.clip_grad_value_(self.parameters(), clip)
            
            model.optimizer.step()
            train_losses.append(loss.data.item())
            train_Rs.append(R)
            
            if verbose and batch_idx%10==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tR: {:.6f}'.format(
                        epoch, batch_idx * train_loader.batch_size, len(train_loader), 100. * batch_idx * float(train_loader.batch_size) / len(train_loader.dataset),
                        loss.item(), R))
                
        if val_loader:
            target_list, pred_list = [], []
            valid_loss, valid_R = 0.0, 0.0
            model.eval()
            
            for batch_idx, batch in enumerate(val_loader):
                seq_X, y = get_input(model.seq_len, batch) 
                out = model(seq_X).view(y.shape)
                loss, R = get_pred_loss(model, seq_X, y)
                
                valid_loss += loss.data.item() #* seq_X.size(0)
                valid_R += R
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
                    save_model(model, modelfile)
            
            
    return {'train_loss':train_losses, 'train_Rs':train_Rs, 'valid_loss':valid_losses, 'valid_Rs':valid_Rs}

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
