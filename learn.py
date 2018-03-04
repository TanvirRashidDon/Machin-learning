# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 04:26:12 2017

@author: Don
"""

import pickle
from sklearn import linear_model

with open('notMNIST1000.pickle','rb') as f:
    data=pickle.load(f)
    
x=data['train_dataset']
size,m,n=x.shape
X=x.reshape(size,-1)

y=data['train_labels']
print('learning....')

lr =linear_model.LogisticRegression()

lr.fit(X, y)




vd=data['valid_dataset']
vset=vd.reshape(vd.shape[0],-1)
vlabels=data['valid_labels']
#lr.predict_proba(vset)
print(lr.predict(vset))

print(lr.score(vset,vlabels))