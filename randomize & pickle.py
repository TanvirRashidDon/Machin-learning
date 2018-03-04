# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 00:18:57 2017

@author: Don
"""
import numpy as np
import os
from six.moves import cPickle as pickle

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)



data_root='.'
pickle_file = os.path.join(data_root, 'notMNIST50.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
  
  
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

from sklearn import linear_model

n_sample_arr = [50, 100, 1000, 5000, len(train_dataset)]

logistic = linear_model.LogisticRegression()

for n_samples in n_sample_arr:
    x_train = np.reshape(train_dataset,(train_size,image_size*image_size))[0:n_samples]
    y_train = train_labels[0:n_samples]

x_valid = np.reshape(valid_dataset,(valid_size,image_size*image_size))
y_valid = valid_labels

x_valid_clean = np.reshape(valid_dataset_clean,(len(valid_dataset_clean),image_size*image_size))
y_valid_clean = valid_labels_clean

x_test = np.reshape(test_dataset,(test_size,image_size*image_size))
y_test = test_labels

x_test_clean = np.reshape(test_dataset_clean,(len(test_dataset_clean),image_size*image_size))
y_test_clean = test_labels_clean

t1 = time.time()

model = logistic.fit(x_train, y_train)

t2 = time.time()

valid_score = model.score(x_valid, y_valid)
valid_score_clean = model.score(x_valid_clean, y_valid_clean)

test_score = model.score(x_test, y_test)
test_score_clean = model.score(x_test_clean, y_test_clean)

print('%d samples LogisticRegression, time taken %0.2fs' % (n_samples,(t2 - t1)))
print('valid score: %f , valid score(clean): %f' % (valid_score,valid_score_clean))
print('test score: %f , test score(clean): %f' % (test_score,test_score_clean))
