# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:31:18 2022

@author: longzheng
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, cosLr, FixedNumpySeed
from oil.tuning.args import argupdated_config
from oil.tuning.study import train_trial
from oil.utils.parallel import try_multigpu_parallelize
from corm_data.collate import collate_fn
from lie_conv.moleculeTrainer import MolecLieResNet, MoleculeTrainer
from oil.datasetup.datasets import split_dataset
import lie_conv.moleculeTrainer as moleculeTrainer
import lie_conv.lieGroups as lieGroups
import functools
import copy
from qm9_like_data import *
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import wandb
import random
import pandas as pd 
import os
from datetime import datetime
import sys
resume = sys.argv[-1] == "--resume"


task='tg' 
device='cuda'
device = torch.device(device)
lr=1e-2 
bs=4
num_epochs=100
network=MolecLieResNet
recenter=False                
subsample=False 
trainer_config={'log_dir':None,'log_suffix':''}
with FixedNumpySeed(0):
    datasets, num_species, charge_scale = Tgdatasets()
    
dataloaders = {key:LoaderTo(DataLoader(dataset,batch_size=bs,num_workers=0,
                shuffle=(key=='train'),pin_memory=False,collate_fn=collate_fn,drop_last=True),
                device) for key,dataset in datasets.items()}
ds_stats = datasets['train'].stats[task]


sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'valid_loss'},
    'parameters':

    {   
        'k':{'values': [128,256,384,512,640]},
        'nbhd':{'values': [50,100,150,200]},
        'num_layers': {'values': [4,5,6,7,8]},
        
      }
}  


sweep_id = wandb.sweep(sweep=sweep_configuration, project='Lieconv')
def main():
    wandb.init(resume=resume)
    net_config = {'k':wandb.config.k,'nbhd':wandb.config.nbhd,'act':'swish','group':lieGroups.SE3(.2),'fill':1/2,'liftsamples':4,
                     'bn':True,'aug':True,'mean':True,'num_layers':wandb.config.num_layers}
    model = network(num_species,charge_scale,**net_config).to(device)
    optimizer = Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience= 5)
    loss_func = nn.MSELoss()

    for epoch in tqdm(range(100)):   
        model.train()
        train_loss = []
        valid_loss = []
        for step,batch in enumerate(dataloaders['train']):
            output = model(batch)
            optimizer.zero_grad()
            loss = loss_func(output,batch['tg'])
            loss.backward()
            optimizer.step()
            temp = loss.cpu()
            temp = temp.detach().numpy().tolist()
            train_loss.append(temp)
        train_loss = np.mean(train_loss)
        scheduler.step(train_loss)


        model.eval()
        valid_loss = []
        for step,batch in enumerate(dataloaders['valid']):
            output = model(batch)
            loss = loss_func(output,batch['tg'])
            temp = loss.cpu()
            temp = temp.detach().numpy().tolist()
            valid_loss.append(temp)
        valid_loss = np.mean(valid_loss)
        wandb.log({"train_loss": train_loss,
                   "valid_loss": valid_loss}
                  )
# Start sweep job.
wandb.agent(sweep_id, function=main, count=40)