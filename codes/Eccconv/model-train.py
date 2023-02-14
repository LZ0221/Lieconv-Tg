# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:01:41 2023

@author: longzheng
"""

import numpy as np
import tensorflow as tf
from class_testdatset import My_Test_DataSet
from tensorflow.data import Dataset
from spektral.data import BatchLoader
from spektral.datasets import QM9
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
import matplotlib.pyplot as plt
import sys

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

epochs =100
batch_size =8

dataset = My_Test_DataSet(amount=None)

idx_tr = []
with open('dataset_train.txt') as f :
    for i in f:
        idx_tr.append(int(i))
        
idx_va = []
with open('dataset_valid.txt') as f :
    for i in f:
        idx_va.append(int(i))
    
idx_te = []
with open('dataset_test.txt') as f :
    for i in f:
        idx_te.append(int(i))

class Net(Model):
    def __init__(self,ecc1=40,ecc2=24,ecc3=24,ds1 = 96,ds2 =256,ds3 = 768,ds4 = 512,ds5 = 64):
        super().__init__()
        self.conv1 = ECCConv(ecc1, activation="relu")
        self.conv2 = ECCConv(ecc2, activation="relu")
        self.conv3 = ECCConv(ecc3, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dense1 = Dense(ds1,activation='relu',kernel_regularizer = 'l2')
        self.dense2 = Dense(ds2,activation='relu',kernel_regularizer = 'l2')
        self.dense3 = Dense(ds3,activation='relu',kernel_regularizer = 'l2')
        self.dense5 = Dense(ds4,activation='relu',kernel_regularizer = 'l2')
        self.dense6 = Dense(ds5,activation='relu',kernel_regularizer = 'l2')
        self.dense8 = Dense(1)

    def call(self, inputs):
        x, a, e = inputs
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.conv3([x, a, e])
        output = self.global_pool(x)
        output = self.dense1(output)
        output = self.dense2(output)
        output = self.dense3(output)
        output = self.dense5(output)
        output = self.dense6(output)
        output = self.dense8(output)
        return output

dataset_tr, dataset_va,dataset_te = dataset[idx_tr], dataset[idx_va],dataset[idx_te]
loader_tr = BatchLoader(dataset_tr, batch_size=batch_size,epochs=epochs)
loader_va = BatchLoader(dataset_va, batch_size=batch_size,epochs=epochs)
   
adam = Adam(lr=0.001)
model = Net()

model.compile(loss='mean_squared_error',optimizer=adam, metrics=['mean_absolute_error'])
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min',min_lr=0.00001)
history = model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, epochs=epochs,
          validation_data=loader_va.load(),validation_steps = loader_va.steps_per_epoch, 
          callbacks=[reduce_lr])

