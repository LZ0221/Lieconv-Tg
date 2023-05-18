# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:33:35 2022

@author: longzheng
"""

import numpy as np
import tensorflow as tf
from ECCdataset import *
from tensorflow.data import Dataset
from spektral.data import BatchLoader
from spektral.layers import ECCConv, GlobalSumPool, GraphMasking
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import tensorflow as tf
from scipy import sparse as sp
from spektral.utils import pad_jagged_array
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from tensorflow.keras.layers import Dense,concatenate,BatchNormalization,LayerNormalization,Dropout,BatchNormalization,Input
from tensorflow.keras.models import Model
import sys
from keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback
import random
import pandas as pd 
resume = sys.argv[-1] == "--resume"

#Importing data to generate datasets
epochs =100
batch_size =8
dataset = Ecc_DataSet(amount=None)


'''
dataset_train.txt: index of training set data.
dataset_valid.txt: index of validation set data.
dataset_test.txt: index of test set data.
'''

idx_train = []
with open('dataset_train.txt') as f :
    for i in f:
        idx_train.append(int(i))
        
idx_valid = []
with open('dataset_valid.txt') as f :
    for i in f:
        idx_valid.append(int(i))
    
idx_test = []
with open('dataset_test.txt') as f :
    for i in f:
        idx_test.append(int(i))
dataset_tr, dataset_va,dataset_te = dataset[idx_train], dataset[idx_valid],dataset[idx_test]

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':

    {   
        'dropout':{'values': [0.1,0.3,0.5,0.7,0.9]},
        'ds1': {'values': [512,1024,1536,2048]},
        'ds2': {'values': [128,256,384,512]},
        'ds3': {'values': [32,64,96,128]},
        'ecc1' : {'values': [16,24,32,40,48,56,64]},
        'ecc2' : {'values': [16,24,32,40,48,56,64]},
        'ecc3' : {'values': [16,24,32,40,48,56,64]}
        
        
      }
}

class Net(Model):
    def __init__(self,ecc1=32,ecc2=32,ecc3=32,ds1 = 64,ds2 =256,ds3 = 512, dropout = 0.5):
        super().__init__()
        self.conv1 = ECCConv(ecc1, activation="relu")
        self.conv2 = ECCConv(ecc2, activation="relu")
        self.conv3 = ECCConv(ecc3, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dropout = Dropout(rate=dropout)
        self.dense1 = Dense(ds1,activation='relu',kernel_regularizer = 'l2')
        self.dense2 = Dense(ds2,activation='relu',kernel_regularizer = 'l2')
        self.dense3 = Dense(ds3,activation='relu',kernel_regularizer = 'l2')
        self.dense4 = Dense(1)

    def call(self, inputs):
        x, a, e = inputs
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.conv3([x, a, e])
        output = self.global_pool(x)
        output = self.dense1(output)
        output = self.dropout(output)
        output = self.dense2(output)
        output = self.dropout(output)
        output = self.dense3(output)
        output = self.dense4(output)
        return output
    
sweep_id = wandb.sweep(sweep=sweep_configuration, project='ECC')
def main():
    loader_tr = BatchLoader(dataset_tr, batch_size=batch_size,epochs=epochs)
    loader_va = BatchLoader(dataset_va, batch_size=batch_size,epochs=epochs)
    wandb.init(resume=resume)
    adam = Adam(lr=0.001)
 
    model = Net(ecc1 = wandb.config.ecc1,ecc2 = wandb.config.ecc2,ecc3 = wandb.config.ecc3,
                ds1 =wandb.config.ds1, ds2 =wandb.config.ds2,ds3 =wandb.config.ds3,
                dropout =wandb.config.dropout)
    model.compile(loss='mean_squared_error',optimizer=adam, metrics=['mean_absolute_error'])
    model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, epochs=epochs,
              validation_data=loader_va.load(),validation_steps = loader_va.steps_per_epoch,
              initial_epoch=wandb.run.step, 
              callbacks=[WandbCallback()])
    
    
    
       
# Start sweep job.
wandb.agent(sweep_id, function=main, count=40)
