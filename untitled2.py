# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 02:43:58 2017

@author: Don
"""

import pickle

def overlapNumber(arrayA,arrayB):
    sum = 0 # number of overlapping rows
    for i in range(arrayA.shape[0]): # iterate over all rows of val_dataset
        overlap = (arrayB == arrayA[i,:,:]).all(axis=1).all(axis=1).sum()
    if overlap:
        sum += 1
    print(sum)
    
    
with open('notMNIST1000.pickle','rb') as f:
    data=pickle.load(f)
    
print('\noverlap among train_dataset and valid_dataset\n')
overlapNumber(data['train_dataset'],data['valid_dataset'])

print('\noverlap among train_dataset and test_dataset\n')
overlapNumber(data['train_dataset'],data['test_dataset'])