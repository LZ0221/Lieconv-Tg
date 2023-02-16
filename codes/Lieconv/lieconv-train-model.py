# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:55:52 2023

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
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

task='tg' 
device='cuda'
device = torch.device(device)

def train(lr,epochs,models_name,bs,recenter=False , subsample=False  ,net_config={'k':256,'nbhd':150,'act':'swish','group':lieGroups.SE3(.2),'fill':1/2,'liftsamples':4,
                'bn':True,'aug':True,'mean':True,'num_layers':8},trainer_config={'log_dir':None,'log_suffix':''}):
    '''
    * Training model
    *
    * Attributes
    * ----------
    * lr             : learning rate
    * epochs         : Number of epochs to iterate over the dataset. By default (None) iterates indefinitely
    * bs             : Size of the mini-batches
    * models_name    : Model save name
    * net_config     : Initialization parameters of lieconv
    * 
    * Returns
    * -------
    * model weight   : Weighting file for the model
    '''
    network=MolecLieResNet
    with FixedNumpySeed(0):
        datasets, num_species, charge_scale = Tgdatasets()
    dataloaders = {key:LoaderTo(DataLoader(dataset,batch_size=bs,num_workers=0,
                    shuffle=(key=='train'),pin_memory=False,collate_fn=collate_fn,drop_last=True),
                    device) for key,dataset in datasets.items()}
    ds_stats = datasets['train'].stats[task]
    
    #load_models
    model = network(num_species,charge_scale,**net_config).to(device)

    optimizer = Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience= 5)
    loss_func = nn.MSELoss()
    train_loss_all = []
    valid_loss_all = []
    mix_valid_loss = 100000
    for epoch in tqdm(range(epochs)):   
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
        print('train_loss',train_loss)
        train_loss_all.append(train_loss)


        model.eval()
        valid_loss = []
        for step,batch in enumerate(dataloaders['valid']):
            output = model(batch)
            loss = loss_func(output,batch['tg'])
            temp = loss.cpu()
            temp = temp.detach().numpy().tolist()
            valid_loss.append(temp)
        valid_loss = np.mean(valid_loss)
        if valid_loss < mix_valid_loss:
            mix_valid_loss = valid_loss
            torch.save(model.state_dict(),models_name)
        else:
            pass
        
        print('valid_loss',valid_loss)
        valid_loss_all.append(valid_loss)
        
def predict( lr,  epochs, models_name,bs,recenter=False , subsample=False  ,net_config={'k':256,'nbhd':150,'act':'swish','group':lieGroups.SE3(.2),'fill':1/2,'liftsamples':4,
                'bn':True,'aug':True,'mean':True,'num_layers':8},trainer_config={'log_dir':None,'log_suffix':''},dataset_name = 'train',):
    '''
    * Evaluation model
    *
    * Attributes
    * ----------
    * lr             : learning rate
    * epochs         : Number of epochs to iterate over the dataset. By default (None) iterates indefinitely
    * bs             : Size of the mini-batches
    * models_name    : Model save name
    * dataset_name   ：Data set to be predicted(train,valid,test)
    * net_config     : Initialization parameters of lieconv
    * 
    * Returns
    * -------
    * y_trues,y_preds : list of true and predicted values for the dataset
    '''
    network=MolecLieResNet
    with FixedNumpySeed(0):
        datasets, num_species, charge_scale = Tgdatasets()
    dataloaders = {key:LoaderTo(DataLoader(dataset,batch_size=bs,num_workers=0,
                    shuffle=(key=='train'),pin_memory=False,collate_fn=collate_fn,drop_last=True),
                    device) for key,dataset in datasets.items()}
    ds_stats = datasets['train'].stats[task]
    #load_models
    model = network(num_species,charge_scale,**net_config).to(device)
    model.load_state_dict(torch.load(models_name))
    model.eval()
    Tg_trues = []
    Tg_preds = []
    for step,batch in enumerate(dataloaders[dataset_name]):
        index = batch['index']
        index = index.cpu()
        index = index.detach().numpy()
        index = index.tolist()
        y_true = batch['tg']
        y_pred = model(batch)
        y_true = y_true.cpu()
        y_true = y_true.detach().numpy()
        y_true = y_true.tolist()
        Tg_trues.append(y_true[0])
        y_pred = y_pred.cpu()
        y_pred = y_pred.detach().numpy()
        y_pred = y_pred.tolist()
        Tg_predsappend(y_pred[0])
        torch.cuda.empty_cache()
    return Tg_trues,Tg_preds

                




