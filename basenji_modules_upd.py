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


class DNA_Iter(Dataset):
    def __init__(self, input_name, chrom_seq, batch_size, chrom, switch=False, target_window = 128):
        self.batch_size = batch_size
        self.chrom = chrom
        self.chrom_seq = chrom_seq
        self.seq = self.read_memmap_input(input_name)
        self.target_window = target_window
        self.nucs = np.arange(5.)
        self.len = (int(self.seq.shape[0] / (2*self.batch_size)))
        self.switch = switch 
        self.switch_func = np.vectorize(lambda x: x + 1 if (x % 2 == 0) else x - 1)
    def __len__(self):
        return self.len 

    def __getitem__(self, idx):      
        seq_subset = self.seq[idx:idx+self.batch_size]
        if self.switch: 
            seq_subset = self.switch_func(seq_subset)
        dta = self.get_csc_matrix(seq_subset)
#         dta = np.transpose(dta)
        tgt = self.seq[self.chrom_seq[self.chrom]+idx:self.chrom_seq[self.chrom]+idx+self.batch_size]
        if (tgt.shape[0] == 0):
            print ('idx', idx, self.chrom, 'chrom seq len', self.chrom_seq[self.chrom], 'self len', self.len, 'seq shape', self.seq.shape[0])
        tgt_window = torch.tensor(self.calc_mean_lst(tgt, self.target_window))
        
#         means = tgt_window.mean(dim=0, keepdim=True)
#         stds = tgt_window.std(dim=0, keepdim=True)
#         normalized_tgt_window = (tgt_window - means) / stds

#         tgt_window = F.normalize(torch.tensor(tgt_window), dim = 0)
        #         tgt_window = F.normalize(torch.tensor(tgt_window))
        return torch.tensor(dta), tgt_window #.view(4, self.batch_size), torch.tensor(tgt_window)

    def read_numpy_input(self, np_gq_name):
        seq = np.load(np_gq_name)
        return seq

    def read_memmap_input(self, mmap_name):
        seq = np.memmap(mmap_name, dtype='float32',  mode = 'r+') #, shape=(2, self.chrom_seq[self.chrom]))
        return seq


    def dna_1hot(self, seq):
        adtensor = np.zeros((4, self.batch_size), dtype=float)
        for nucidx in range(len(self.nucs)):
            nuc = self.nucs[nucidx]#.encode()
            j = np.where(seq[0:len(seq)] == nuc)[0]
            adtensor[nucidx, j] = 1
        return adtensor

    def get_csc_matrix(self, seq_subset):
        N, M = len(seq_subset), len(self.nucs)
        dtype = np.uint8
        rows = np.arange(N)
        cols = seq_subset
        data = np.ones(N, dtype=dtype)
        ynew = csc_matrix((data, (rows, cols)), shape=(N, M), dtype=dtype)
        return ynew.toarray()[:, :4]

    def calc_mean_lst(self, lst, n):
        return np.array([np.mean(lst[i:i + n]) for i in range(int(len(lst)/n))])
def regularize_loss(modelparams, net, loss):
    lambda1 = modelparams["lambda_param"]
    ltype = modelparams["ltype"]
    if ltype == 3:
            torch.nn.utils.clip_grad_norm_(
                net.conv_block_1.parameters(), lambda1)
            torch.nn.utils.clip_grad_norm_(
                net.conv_block_2.parameters(), lambda1)
            torch.nn.utils.clip_grad_norm_(
                net.conv_block_3.parameters(), lambda1)
            torch.nn.utils.clip_grad_norm_(
                net.conv_block_4.parameters(), lambda1)
            torch.nn.utils.clip_grad_norm_(
                    net.conv_block_5.parameters(), lambda1)
            for i in range(len(net.dilations)):
                torch.nn.utils.clip_grad_norm_(
                    net.dilations[i].parameters(), lambda1)
    
    else:      
        l0_params = torch.cat(
            [x.view(-1) for x in net.conv_block_1[1].parameters()])
        l1_params = torch.cat(
            [x.view(-1) for x in net.conv_block_2[1].parameters()])
        l2_params = torch.cat(
            [x.view(-1) for x in net.conv_block_3[1].parameters()])
        l3_params = torch.cat(
            [x.view(-1) for x in net.conv_block_4[1].parameters()])
        l4_params = torch.cat(
                [x.view(-1) for x in net.conv_block_5[1].parameters()])
        dil_params = []
        for i in range(len(net.dilations)):
            dil_params.append(torch.cat(
                [x.view(-1) for x in net.dilations[i][1].parameters()]))
        
    if ltype in [1, 2]:
        l1_l0 = lambda1 * torch.norm(l0_params, ltype)
        l1_l1 = lambda1 * torch.norm(l1_params, ltype)
        l1_l2 = lambda1 * torch.norm(l2_params, ltype)
        l1_l3 = lambda1 * torch.norm(l3_params, ltype)
        l1_l4 = lambda1 * torch.norm(l4_params, 1)
        l1_l4 = lambda1 * torch.norm(l4_params, 2)
        dil_norm = []
        for d in dil_params:
            dil_norm.append(lambda1 * torch.norm(d, ltype))  
        loss = loss + l1_l0 + l1_l1 + l1_l2 + l1_l3 + l1_l4 + torch.stack(dil_norm).sum()
    
    elif ltype == 4:
        l1_l0 = lambda1 * torch.norm(l0_params, 1)
        l1_l1 = lambda1 * torch.norm(l1_params, 1)
        l1_l2 = lambda1 * torch.norm(l2_params, 1)
        l1_l3 = lambda1 * torch.norm(l3_params, 1)
        l2_l0 = lambda1 * torch.norm(l0_params, 2)
        l2_l1 = lambda1 * torch.norm(l1_params, 2)
        l2_l2 = lambda1 * torch.norm(l2_params, 2)
        l2_l3 = lambda1 * torch.norm(l3_params, 2)
        l1_l4 = lambda1 * torch.norm(l4_params, 1)
        l2_l4 = lambda1 * torch.norm(l4_params, 2)
        dil_norm1, dil_norm2 = [], []
        for d in dil_params:
            dil_norm1.append(lambda1 * torch.norm(d, 1))  
            dil_norm2.append(lambda1 * torch.norm(d, 2))  

        loss = loss + l1_l0 + l1_l1 + l1_l2 +\
                l1_l3 + l1_l4 + l2_l0 + l2_l1 +\
                l2_l2 + l2_l3 + l2_l4 + \
            torch.stack(dil_norm1).sum() + torch.stack(dil_norm2).sum()
    return loss
def get_train_val_loader(input_files_dir, chrom_seq, batch_size, switch):
    
#     torch.backends.cudnn.deterministic = True
#     random.seed(1)
#     torch.manual_seed(1)
#     torch.cuda.manual_seed(1)
    np.random.seed(1)
    input_files = [file for file in os.listdir(memmap_data_contigs_dir) if file.split('.')[-1] == 'dta']    
    np.random.shuffle(input_files)
    input_chroms = [input_file.split('_')[0] for input_file in input_files]
    valid_dset = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, input_files[i]), chrom_seq, 128, input_chroms[i]) for i in range(0, 1)])
    training_dset = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, input_files[i]), chrom_seq, 128, input_chroms[i], switch) for i in range(1, 15)])
    train_loader = DataLoader(dataset=training_dset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=10, pin_memory=True) #, num_workers=4)
    val_loader = DataLoader(dataset=valid_dset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=8, pin_memory=True) #, num_workers=4)
    return train_loader, val_loader


def _get_conv1d_out_length(l_in, padding, dilation, kernel, stride):
    return int((l_in + 2*padding - dilation*(kernel-1) - 1)/stride)+1



def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def pearsonr_pt(x, y):
    mean_x, mean_y= torch.mean(x), torch.mean(y)
    xm, ym = x.sub(mean_x), y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    r2 = r2_loss(x, y)
    return r_val, r2

def get_input(batch): 
    seq_X,y = batch
    X_reshape = torch.stack(torch.chunk(torch.transpose(seq_X.reshape(seq_X.shape[0]*seq_X.shape[1], 4), 1, 0), 4, dim=1)).type(torch.FloatTensor).cuda(2)
    y =  y.type(torch.FloatTensor).cuda(2)
    return X_reshape, y

def get_pred_loss(model, seq_X, y, reverse_switch=False):
    out = model(seq_X).view(y.shape)
    if reverse_switch: 
        out = torch.flip(torch.flatten(out), [0]).view(y.shape)
    modelparams = {"lambda_param": 0., "ltype":2}
    loss = model.loss_fn(out,y)
    loss = regularize_loss(modelparams, model, loss)
    out = torch.flatten(out.float()).to('cpu').detach().float()
    y = torch.flatten(y.float()).to('cpu').detach().float()
    R, r2 = pearsonr_pt(out, y)
    return loss, R, r2, out

def train_model(model, epochs=100, clip=0.75, device='cpu', modelfile='models/best_ret_checkpoint.pt', logfile = None, tuneray=False, verbose=True, debug=False):
    train_losses, valid_losses, train_Rs, valid_Rs = [], [], [], []
    best_model=-1
    scaler = GradScaler()
    for epoch in range(epochs):
#         if epoch % 2 == 0: 
#             switch = True
#         else: 
#             switch = False
        switch=False
        train_loader, val_loader = get_train_val_loader(memmap_data_contigs_dir, chrom_seq, int(model.seq_len/128), switch)
        print (len(train_loader), len(val_loader))
        for batch_idx, batch in enumerate(train_loader):
            num_samples_iter = int(len(train_loader.sampler) / len(train_loader))
#             if batch_idx == 2:
#                 break
            model.train()
            model.optimizer.zero_grad()
            seq_X, y = get_input(batch)
#             print (np.unique(y.flatten().cpu().numpy()))
            if debug: 
                print ('X', seq_X.shape, 'y', y.shape)

            if seq_X.shape[-1] == int(model.seq_len / 4): 
                if debug: 
                    print ('X', seq_X.shape, 'y', y.shape)
                with autocast():
                    loss, R, r2, out = get_pred_loss(model, seq_X, y, switch)
#                     train_loss += loss.item()*data.size(0)
#                     running_loss += loss.item()
#                     print (loss, len(train_loader.sampler))
                    if batch_idx % 10 == 0: 
#                         print (out[0][:10].cpu().detach().numpy())
                        print (torch.sum(out).item(), torch.sum(y).item())
#                 loss.sum().backward()

#                     loss = loss/len(train_loader.sampler)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
#                 torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
              
#                 model.optimizer.step()
                scaler.step(model.optimizer)
#                 model.scheduler.step()
                scaler.update()
                train_losses.append(loss.data.item())
                train_Rs.append(R)
                if verbose and batch_idx%10==0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tR: {:.6f}\tR2: {:.6f}'.format(
                          epoch, int(batch_idx * train_loader.batch_size/int(model.seq_len/128)), len(train_loader), 100. * batch_idx * float(train_loader.batch_size) / len(train_loader.dataset),
                          loss.item(), R, r2))
            else: 
                continue
            if batch_idx % 300 == 0:
                ys = y.flatten().cpu().numpy()
                preds = out.flatten().detach().cpu().numpy()
                plt.plot(np.arange(len(ys.flatten())), ys.flatten(), label='True')
                plt.plot(np.arange(len(preds.flatten())), preds.flatten(), label='Predicted', alpha=0.5)
                plt.legend()
                plt.show()

        print ('batch_idx ', batch_idx)
        print ('Mean training loss: ', np.mean(train_losses[epoch*batch_idx:(1+epoch)*batch_idx]) )
        print ('Mean R: ', np.mean(train_Rs[epoch*batch_idx:(1+epoch)*batch_idx]) )


        if val_loader:
            num_samples_iter_val = int(len(val_loader.sampler) / len(val_loader))
            target_list, pred_list = [], []
            valid_loss, valid_R = 0.0, 0.0
            model.eval()
            
            for batch_idx, batch in enumerate(val_loader):
#                 if batch_idx == 1:
#                     break
                seq_X, y = get_input(batch) 
                if seq_X.shape[-1] == int(model.seq_len / 4):
                    out = model(seq_X).view(y.shape)
                    loss, R, r2, out = get_pred_loss(model, seq_X, y, switch)
                    valid_loss += loss.data.item() #* seq_X.size(0)
                    if verbose and batch_idx%10==0:
                        print('Validation: Loss: {:.6f}\tR: {:.6f}'.format(loss.data.item(), R))
                    valid_R += R
                else: 
                    continue
            valid_Rs.append(valid_R / len(val_loader)) #.dataset))
            valid_losses.append(valid_loss / len(val_loader)) #.dataset))
#             model.scheduler.step()
            if tuneray:
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), model.optimizer.state_dict()), path)
                tune.report(loss=valid_loss, corel=valid_R)

#             if verbose:
#                 print('Validation: Loss: {:.6f}\tR: {:.6f}'.format(valid_loss/ len(val_loader), valid_R / len(val_loader))
            
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