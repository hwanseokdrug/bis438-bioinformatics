# install rdkit library on colab environment
!wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!time bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
!time conda install -q -y -c conda-forge rdkit


# %matplotlib inline
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/usr/local/lib/python3.7/site-packages/')


#import basic python packages
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from rdkit import rdBase

import tensorflow as tf
import tensorflow.keras as keras
from keras import Model
from keras.backend import one_hot
from keras.layers import Activation, Dense, Dropout, Input
from keras.utils import np_utils


# load data from files
from google.colab import files
data_AR = files.upload() # select 'nr-ar.smiles' file and upload it
data_ER = files.upload() # select 'nr-er.smiels' file and upload it


# save the uploaded data as Pandas dataframe
import io
df_AR = pd.read_csv(io.BytesIO(data_AR['nr-ar.smiles']), sep = '\t', names = ['smiles', 'array', 'activity'])
df_ER = pd.read_csv(io.BytesIO(data_ER['nr-er.smiles']), sep = '\t', names = ['smiles', 'array', 'activity'])

smi_AR = df_AR['smiles']
smi_ER = df_ER['smiles']
activity_AR = df_AR['activity']
activity_ER = df_ER['activity']


# data preprocessing; convert SMILES string to 2048 bit Fingerprint
def preprocessing(data_X, data_Y):
###################
# Your code here
###################
  return np.array(data_x), np.array(data_y)

data_X_AR, data_Y_AR = preprocessing(smi_AR, activity_AR)
data_X_ER, data_Y_ER = preprocessing(smi_ER, activity_ER)

# check the shape of the raw data
print(data_X_AR.shape)
print(data_Y_AR.shape)
print(data_X_ER.shape)
print(data_Y_ER.shape)


# split the training, validation and test dataset. train:validation:test=8:1:1
from sklearn.model_selection import train_test_split
#split training and test dataset from raw dataset
###################
# Your code here
###################

#split training and validation dataset from training dataset
###################
# Your code here
###################

# check the shape of the splited data
print(X_train_AR.shape, Y_train_AR.shape)
print(X_valid_AR.shape, Y_valid_AR.shape)
print(X_test_AR.shape, Y_test_AR.shape)
print(X_train_ER.shape, Y_train_ER.shape)
print(X_valid_ER.shape, Y_valid_ER.shape)
print(X_test_ER.shape, Y_test_ER.shape)


# build a neural network model
# the activation function of hidden layer is relu, and output is softmax

###################
# Your code here
###################

# make a prediction model for AR
###################
# Your code here
###################
model_AR.summary() # print summary of AR model


# make a predictin model for ER
###################
# Your code here
###################
model_ER.summary()# print summary of ER model


# early stopping with validation dataset 
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

es_AR = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc_AR = ModelCheckpoint('best_model_AR.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
es_ER = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc_ER = ModelCheckpoint('best_model_ER.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# training the model using training and validation set
epochs = 50
hist_AR = model_AR.fit(X_train_AR, Y_train_AR, validation_data=(X_valid_AR, Y_valid_AR), batch_size=32, epochs=epochs, verbose=0, callbacks=[es_AR, mc_AR])
hist_ER = model_ER.fit(X_train_ER, Y_train_ER, validation_data=(X_valid_ER, Y_valid_ER),batch_size=32, epochs=epochs, verbose=0, callbacks=[es_ER, mc_ER])

# plot the accuracy and loss value of each model.
###################
# Your code here
###################


# evaluate the prediction model using test dataset.
from sklearn.metrics import classification_report, roc_curve, auc
# AUC of AR dataset
###################
# Your code here
###################

# AUC of ER dataset
###################
# Your code here
###################
