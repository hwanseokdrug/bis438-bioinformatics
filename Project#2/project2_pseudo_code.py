#### project2_pseudo_code.py

# import
import os
import numpy as np
import random

import keras
import tensorflow as tf
###################
# Import keras modules here
###################


# import my google drive
from google.colab import drive
drive.mount('/content/gdrive')

# read fasta files  
fasta_file_path='gdrive/My Drive/data sample/'
rbp_name = 'ELAVL1'
print('list of files', os.listdir(fasta_file_path))

# loading training set
train_fasta = list()
train_label = list()
for single_file in [x for x in os.listdir(fasta_file_path) if rbp_name in x and 'train' in x]:
  print('Processing file...', single_file)
  with open(fasta_file_path + single_file) as f:
    for line in f.readlines():
      # get fasta sequence
      if '>' in line:
        continue
      else:
        train_fasta.append(line.strip())
      # get positive negative label
      if 'positives' in single_file:
        train_label.append(1)
      else:
        train_label.append(0)
        

# convert sequence file to one-hot encoding representation
# function: fasta to onehot representation
def convert2onehot(sequence_list):
  map = {
      'A':[1,0,0,0],
      'U':[0,1,0,0],
      'T':[0,1,0,0],
      'G':[0,0,1,0],
      'C':[0,0,0,1]
  }
  
  onehot = []
  for single_sequence in sequence_list:
    single_onehot = []
    for x in single_sequence:
      single_onehot.append(map[x.upper()])
    onehot.append(single_onehot)
    
  return np.asarray(onehot, dtype=np.float32)

data_input = convert2onehot(train_fasta)
data_label = keras.utils.to_categorical(train_label, 2)

# split training set into training set and validation set
# random shuffling of training data
###################
# Your code here
###################
print('Dataset preparation done... train_input, train_label, validation_input, validation_label')
print('Size of each set...', train_input.shape, train_label.shape, validation_input.shape, validation_label.shape)


# model building
###################
# Your code here
###################

# model training
###################
# Your code here
###################

# evaluation: calculate AUC score for the test set
# loading test sets 
###################
# Your code here
###################
            
test_input = convert2onehot(test_fasta)
test_label = keras.utils.to_categorical(test_label, 2)
print('TEST set prepared...', test_input.shape, test_label.shape)


# calculate AUC score for test sets
###################
# Your code here
###################