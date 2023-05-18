# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:33:35 2022

@author: longzheng
"""

import numpy as np
import tensorflow as tf
from class_testdatset import My_Test_DataSet
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
epochs =100
batch_size =8

resume = sys.argv[-1] == "--resume"


dataset = My_Test_DataSet(amount=None)
idxs = np.random.permutation(len(dataset))
split_1 = int(0.8 * len(dataset))
split_2 = int(0.1 * len(dataset))
idx_tr, temp = np.split(idxs, [split_1])
idx_va, idx_te = np.split(temp, [split_2])
dataset_tr, dataset_va,dataset_te = dataset[idx_tr], dataset[idx_va],dataset[idx_te]

f = open('dataset_train.txt','a')
for i in idx_tr:
    f.write(str(i))
    f.write('\n')
f.close()

f = open('dataset_valid.txt','a')
for i in idx_va:
    f.write(str(i))
    f.write('\n')
f.close()

f = open('dataset_test.txt','a')
for i in idx_te:
    f.write(str(i))
    f.write('\n')
f.close()

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':

    {   
        'dropout':{'values': [0.1,0.3,0.5,0.7,0.9]},
        'learn_rate':{'values': [0.01,0.001,0.0005]},
        'ds1': {'values': [512,1024,1536,2048]},
        'ds2': {'values': [128,256,384,512]},
        'ds3': {'values': [32,64,96,128]},
        # 'ds4': {'values': [128,256,384,512]},
        # 'ds5': {'values': [48,64,80,96]},
        'ecc1' : {'values': [16,24,32,40,48,56,64]},
        'ecc2' : {'values': [16,24,32,40,48,56,64]},
        'ecc3' : {'values': [16,24,32,40,48,56,64]}
        # 'fliter1': {'values': [4,6,8,10,12,16,18,20]},
        # 'fliter2': {'values': [4,6,8,10,12,16]},
        # 'ws1': {'values': [8,10,12,16,20]},
        # 'ws2': {'values': [ 2,4,6,8 ]},
        
        
      }
}

class Net(Model):
    def __init__(self,ecc1=32,ecc2=32,ecc3=32,ds1 = 64,ds2 =256,ds3 = 512, dropout = 0.5):
        super().__init__()
        # self.masking = GraphMasking()
        self.conv1 = ECCConv(ecc1, activation="relu")
        self.conv2 = ECCConv(ecc2, activation="relu")
        self.conv3 = ECCConv(ecc3, activation="relu")
        # self.conv4 = ECCConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        # self.ln = LayerNormalization()
        self.dropout = Dropout(rate=dropout)
        self.dense1 = Dense(ds1,activation='relu',kernel_regularizer = 'l2')
        self.dense2 = Dense(ds2,activation='relu',kernel_regularizer = 'l2')
        self.dense3 = Dense(ds3,activation='relu',kernel_regularizer = 'l2')
        # self.dense4 = Dense(512,activation='relu',kernel_regularizer = 'l2')
        # self.dense5 = Dense(ds4,activation='relu',kernel_regularizer = 'l2')
        # self.dense6 = Dense(ds5,activation='relu',kernel_regularizer = 'l2')
        # self.dense7 = Dense(32,activation='relu',kernel_regularizer = 'l2')
        self.dense4 = Dense(1)

    def call(self, inputs):
        x, a, e = inputs
        # x = self.masking(x)
        
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.conv3([x, a, e])
        # x = self.conv4([x, a, e])
        output = self.global_pool(x)
        # output = concatenate([output, Mw],axis=1)
        output = self.dense1(output)
        output = self.dropout(output)
        output = self.dense2(output)
        output = self.dropout(output)
        output = self.dense3(output)
        # output = self.dense5(output)
        # output = self.dense6(output)
        # output = self.dense6(output)
        # output = self.dense7(output)
        output = self.dense4(output)

        return output
sweep_id = wandb.sweep(sweep=sweep_configuration, project='Ecc_hyperparameter_3_40')
def main():
    loader_tr = BatchLoader(dataset_tr, batch_size=batch_size,epochs=epochs)
    loader_va = BatchLoader(dataset_va, batch_size=batch_size,epochs=epochs)
    wandb.init(resume=resume)
    adam = Adam(lr=wandb.config.learn_rate)
 
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
