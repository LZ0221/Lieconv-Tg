# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:43:07 2023

@author: longzheng
"""

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
import pandas as pd 
import deepchem as dc
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau

# Importing Data
df = pd.read_csv('data.csv',encoding= "unicode_escape")
df_number_smile_Tg = df.values[:]

index_list = []
PID_list = []
tg_list = []
smile_list = []
for i in range(len(df_number_smile_Tg)):
    temp0,temp1,temp2,temp3 = df_number_smile_Tg[i]
    index_list.append(temp0)
    PID_list.append(temp1)
    tg_list.append(temp2)
    smile_list.append(temp3)

# Generate datasets
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

encodings_list = []
targets_list = []
featurizer = dc.feat.OneHotFeaturizer(charset = ['*', 'c', '1', '(', 'C', '2', 'N', '3', '=', 'O', ')', '4', '5', '6', '7', '8', 'F', 'n', 'S', '-', 's', '[', '@', 'H', ']', 'i', 'o', '9', '%', '0', 'P', '+', '#', 'l', 'B', 'r'],max_length = 250)
for num in idx_tr:
    temp=[]
    smiles = smile_list[num]
    temp.append(smiles) 
    encodings = featurizer.featurize(temp)
    a,b,c = encodings.shape
    encodings = encodings.reshape((b,c,1))
    encodings_list.append(encodings)
    target = tg_list[num]
    targets_list.append(target)

train_X = np.array(encodings_list)
train_Y =  np.array(targets_list)

 
encodings_list = []
targets_list = []  
for num in idx_va:
    temp=[]
    smiles = smile_list[num]
    temp.append(smiles) 
    encodings = featurizer.featurize(temp)
    a,b,c = encodings.shape
    encodings = encodings.reshape((b,c,1))
    encodings_list.append(encodings)
    target = tg_list[num]
    targets_list.append(target)
    
Valid_X = np.array(encodings_list)
Valid_Y =  np.array(targets_list)

encodings_list = []
targets_list = []     
for num in idx_te:
    temp=[]
    smiles = smile_list[num]
    temp.append(smiles) 
    encodings = featurizer.featurize(temp)
    a,b,c = encodings.shape
    encodings = encodings.reshape((b,c,1))
    encodings_list.append(encodings)
    target = tg_list[num]
    targets_list.append(target)
    
Test_X = np.array(encodings_list)
Test_Y =  np.array(targets_list)

# train model
img_width, img_height = 250, 37
X_train = train_X.astype('float32') 
X_valid = Valid_X.astype('float32') 
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min',min_lr=0.00001)
mcp_save = ModelCheckpoint('2023_2_7_20.54_36/image-cnn-36.hdf5', save_best_only=True, monitor='val_loss', mode='min')
adam = Adam(lr=0.001)

model = Sequential()
model.add(Conv2D(18, (8, 8), activation='relu',
                 input_shape=(img_width, img_height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu')) 
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer=adam, metrics=['mean_absolute_error'])
history = model.fit(X_train, train_Y,  validation_data=(X_valid, Valid_Y),
          epochs=100, 
          batch_size = 4 ,
          callbacks=[reduce_lr_loss,mcp_save])
