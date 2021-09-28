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