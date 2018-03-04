# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 05:54:18 2017

@author: Don
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

#load data
pickle_file = 'data.pickle'
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  
  
  #resize data
  
image_size = 28
num_labels = 10
beta = 5e-4
batch_size = 2000
initial_learning_rate=0.001

num_steps = 1001

hidden_layer1_size = 8192
hidden_layer2_size =4096
hidden_layer3_size = 1024
stddev=0.1

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# helper functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=stddev)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(stddev, shape=shape)
  return tf.Variable(initial)


#initialize tensorflow graph for sgd
graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Variables.
    hidden1_weights = weight_variable([image_size * image_size, hidden_layer1_size])
    hidden1_biases= bias_variable([hidden_layer1_size])
    hidden1_layer = 2*tf.nn.dropout( tf.nn.relu(tf.matmul(tf_train_dataset, hidden1_weights) + hidden1_biases),.5)

    hidden2_weights = weight_variable([hidden_layer1_size, hidden_layer2_size])
    hidden2_biases = bias_variable([hidden_layer2_size])
    hidden2_layer = 2*tf.nn.dropout(tf.nn.relu(tf.matmul(hidden1_layer, hidden2_weights) + hidden2_biases),.5)

    hidden3_weights = weight_variable([hidden_layer2_size, hidden_layer3_size])
    hidden3_biases = bias_variable([hidden_layer3_size])
    hidden3_layer = 2*tf.nn.dropout(tf.nn.relu(tf.matmul(hidden2_layer, hidden3_weights) + hidden3_biases),.5)


    output_weights = weight_variable([hidden_layer3_size, num_labels])
    output_biases = bias_variable([num_labels])
    
    # Training computation.    
    logits = tf.matmul(hidden3_layer, output_weights) + output_biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    #regularization
    L2_regul = tf.nn.l2_loss(hidden1_weights) + tf.nn.l2_loss(hidden1_biases) + tf.nn.l2_loss(hidden2_weights) + tf.nn.l2_loss(
        hidden2_biases)+tf.nn.l2_loss(hidden3_weights)+tf.nn.l2_loss(hidden3_biases)
    loss=loss+beta*L2_regul

    # Optimizer.
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,decay_steps=50, decay_rate=0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training.
    train_prediction = tf.nn.softmax(logits)
    
    # Setup validation prediction step.        
    valid_hidden1 = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden1_weights) + hidden1_biases)
    valid_hidden2 = tf.nn.relu(tf.matmul(valid_hidden1, hidden2_weights) + hidden2_biases)
    valid_hidden3 = tf.nn.relu(tf.matmul(valid_hidden2,hidden3_weights)+hidden3_biases)
    valid_logits = tf.matmul(valid_hidden3, output_weights) + output_biases
    valid_prediction = tf.nn.softmax(valid_logits)

    # And setup the test prediction step.
    test_hidden1 = tf.nn.relu(tf.matmul(tf_test_dataset, hidden1_weights) + hidden1_biases)
    test_hidden2 = tf.nn.relu(tf.matmul(test_hidden1, hidden2_weights) + hidden2_biases)
    test_hidden3 = tf.nn.relu(tf.matmul(test_hidden2,hidden3_weights)+hidden3_biases)
    test_logits = tf.matmul(test_hidden3, output_weights) + output_biases
    test_prediction = tf.nn.softmax(test_logits)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 10 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
