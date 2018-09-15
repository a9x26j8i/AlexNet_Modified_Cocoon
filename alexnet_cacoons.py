# -*- coding: utf-8 -*-

""" AlexNet.

Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.

Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

"""

from __future__ import division, print_function, absolute_import
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

##import tflearn.datasets.oxflower17 as oxflower17
##X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

import tensorflow as tf
import tflearn 
import h5py
from tflearn.data_utils import build_hdf5_image_dataset

##Building Dataset
root_folder='./cacoons/jpg/'
print('-----------Start making dataset!----------')
if not os.path.exists('cacoon.h5'):
	build_hdf5_image_dataset(root_folder,
	image_shape=(227,227),
	mode='folder', 
	output_path='cacoon.h5', 
	categorical_labels=True, 
	normalize=False)
print('------------Dataset is prepared!-------------- ')
h5f=h5py.File('cacoon.h5', 'r')
X=h5f['X']
Y=h5f['Y']

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 48, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 128, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 192, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 2048, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2048, activation='relu')##
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')

network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)


# Training
model = tflearn.DNN(network, checkpoint_path='cacoon.tfl.ckpt',
                    max_checkpoints=3, tensorboard_verbose=2, 
		    best_checkpoint_path='cacoon-best.tfl.ckpt',
		    tensorboard_dir='./tensorboard/logs/')
model.fit(X, Y, n_epoch=200, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=256, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_cacoons')


