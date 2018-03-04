# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 00:25:07 2017

@author: Don
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

def displayImage(folder):
    image_files = os.listdir(folder)
    for image in image_files:
        try:
            image_file = os.path.join(folder, image)
            if image_file.size==None:
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            display(image_file)
            
        except IOError as e:
               print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')