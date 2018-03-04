# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:02:54 2017

@author: panda
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import display, Image
from scipy import ndimage
from six.moves import cPickle as pickle

image_size = 100  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

# display(Image(filename="train_set/judo/7857 resized.png"))

train_folder_path = os.path.join('.', "train_set")
test_folder_path = os.path.join('.', "test_set")

train_folders = []
for fn in os.listdir(train_folder_path):
    train_folders.append(os.path.abspath(os.path.join(train_folder_path, fn)))
test_folders = []
for fn in os.listdir(test_folder_path):
    test_folders.append(os.path.abspath(os.path.join(test_folder_path, fn)))


def load_letter(folder):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = ndimage.imread(image_file)
            image_data = (np.dot(image_data[...,:3], [0.299, 0.587, 0.114])-
                          pixel_depth / 2) / pixel_depth    #rgb to gray

            if image_data.shape != (image_size, image_size):
                print('shape didn''t match:')
                continue
                # raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = maybe_pickle(train_folders)
test_datasets = maybe_pickle(test_folders)

pickle_file = train_datasets[0]  # index 0 should be all As, 1 = all Bs, etc.
with open(pickle_file, 'rb') as f:
    letter_set = pickle.load(f)  # unpickle
    sample_idx = np.random.randint(len(letter_set))  # pick a random image index
    sample_image = letter_set[sample_idx, :, :]  # extract a 2D slice
    plt.figure()
    plt.imshow(sample_image)

