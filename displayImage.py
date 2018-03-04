# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:22:39 2017

@author: Don
"""

import os,IPython,random,pickle
import matplotlib.pyplot as plt
folder=r'D:\cFile\myWork\machin learning\notMNIST_large\A'
file='1.png'

fileName=os.path.join(folder,file)

IPython.display.Image(filename=fileName)

random_filename = random.choice([
    x for x in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, file))
])

rf=os.path.join(folder,random_filename)
IPython.display.Image(filename=rf)


"""
display from pickle

"""

folder=r'D:\cFile\myWork\machin learning\notMNIST_large'
file='A.pickle'

fpath=os.path.join(folder,file)

f=open(fpath,'rb')

dataA=pickle.load(f)
f.close()
plt.imshow(dataA[1,:,:])