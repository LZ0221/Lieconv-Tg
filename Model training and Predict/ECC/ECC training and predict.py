# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:01:41 2023

@author: longzheng
"""

from ECCdataset import Ecc_DataSet
from spektral.data import BatchLoader
from spektral.layers import ECCConv, GlobalSumPool, GraphMasking
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,concatenate,BatchNormalization,LayerNormalization,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def load_dataset(Dataset):
    '''
    * Import the dataset and divide it into training, 
    * validation and test sets in a ratio of 8 : 1 : 1.
    *
    * Attributes
    * ----------
    * Dataset  : datasets to be divided
    * 
    * Returns
    * -------
    * dataset_tr, dataset_va,dataset_te : three scaled data sets
    '''
    
    idxs = np.random.permutation(len(Dataset))
    split_1 = int(0.8 * len(Dataset))
    split_2 = int(0.1 * len(Dataset))
    idx_tr, temp = np.split(idxs, [split_1])
    idx_va, idx_te = np.split(temp, [split_2])
    dataset_tr, dataset_va,dataset_te = Dataset[idx_tr], Dataset[idx_va],Dataset[idx_te]
    return dataset_tr, dataset_va,dataset_te

class Ecc_model(Model):
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

def train(learning_rate,epochs,bs):
    '''
    * Training model
    *
    * Attributes
    * ----------
    * learning_rate  : learning rate
    * epochs         : Number of epochs to iterate over the dataset. By default (None) iterates indefinitely
    * bs             : Size of the mini-batches
    * 
    * Returns
    * -------
    * model : The constructed model
    '''
    dataset = Ecc_DataSet(amount=None)
    dataset_tr, dataset_va,dataset_te = load_dataset(dataset)
    loader_tr = BatchLoader(dataset_tr, batch_size=bs,epochs=epochs)
    loader_va = BatchLoader(dataset_va, batch_size=bs,epochs=epochs)
    model = Ecc_model()
    adam = Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error',optimizer=adam, metrics=['mean_absolute_error'])
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min',min_lr=0.00001)
    history = model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, epochs=epochs,
              validation_data=loader_va.load(),validation_steps = loader_va.steps_per_epoch, 
              callbacks=[reduce_lr])
    model_weight_path = "Saved_model/"
    model.save_weights(model_weight_path)
    return model

def predict(model_weight_path,dataset):
    '''
    * Evaluation model
    *
    * Attributes
    * ----------
    * model_weight_path  : save path of the model
    * dataset            : data set to be predicted
    * 
    * Returns
    * -------
    * y_trues,y_preds : list of true and predicted values for the dataset
    '''
    adam = Adam(lr=0.001)
    data_loader = BatchLoader(dataset, batch_size=1,epochs=1)
    model = Ecc_model()
    model.compile(loss='mean_squared_error',optimizer=adam, metrics=['mean_absolute_error'])
    model.load_weights(model_weight_path)
    y_trues = []
    y_preds = []
    for i in data_loader:
        inputs, target = i
        target1 = target[:,0:1]
        target1 = float(target1.tolist()[0][0])
        y_trues.append(target1)
        yp = model(inputs)
        yp = float(yp[0])
        y_preds.append(yp)
    return y_trues,y_preds
    