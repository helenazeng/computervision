
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import pylab
import mahotas as mh


# In[2]:

color = []
mask = []
normal = []
for i in range(0,20000):
    color.append(mh.imread('project/eecs442challenge/train/color/'+str(i)+'.png'))
    mask.append(mh.imread('project/eecs442challenge/train/mask/'+str(i)+'.png'))
    normal.append(mh.imread('project/eecs442challenge/train/normal/'+str(i)+'.png'))


# In[ ]:

test = color[0]
pylab.imshow(test)
pylab.show()
print(color[100].shape)
mh.imsave(str(i)+'.png',test)


# In[ ]:

print(len(color))
print(len(normal))

print(color[0].shape)
print(normal[0].shape)


# In[ ]:

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Data loading and preprocessing
#import tflearn.datasets.oxflower17 as oxflower17
#X, Y = oxflower17.load_data(one_hot=True)
X = color
Y = normal

# TODO: apply masks

# Building 'VGG Network'
network = input_data(shape=[None, 128, 128, 3])

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 128*128*3, activation='softmax')
network = tflearn.reshape(network, [-1, 128, 128, 3])

network = regression(network, optimizer='momentum',
                     loss='mean_square',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=2)
# tensorboard_verbose from 0-3 3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.
# check point default = none
model.fit(X, Y, n_epoch=3, shuffle=True,
          show_metric=True, batch_size=20, snapshot_step=500,
          snapshot_epoch=False, run_id='project_test')
#plt.imshow(mh.as_rgb(red, green, blue))


# In[ ]:

from __future__ import print_function

import tensorflow as tf
import tflearn

# --------------------------------------
# High-Level API: Using TFLearn wrappers
# --------------------------------------

# Using MNIST Dataset
import tflearn.datasets.mnist as mnist
mnist_data = mnist.read_data_sets(one_hot=True)

# User defined placeholders
with tf.Graph().as_default():
    # Placeholders for data and labels
    X = tf.placeholder(shape=(None, 784), dtype=tf.float32)
    Y = tf.placeholder(shape=(None, 10), dtype=tf.float32)

    net = tf.reshape(X, [-1, 28, 28, 1])

    # Using TFLearn wrappers for network building
    net = tflearn.conv_2d(net, 32, 3, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.local_response_normalization(net)
    net = tflearn.dropout(net, 0.8)
    net = tflearn.conv_2d(net, 64, 3, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.local_response_normalization(net)
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 128, activation='tanh')
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 256, activation='tanh')
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 10, activation='linear')

    # Defining other ops using Tensorflow
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        batch_size = 128
        for epoch in range(2): # 2 epochs
            avg_cost = 0.
            total_batch = int(mnist_data.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
                cost = sess.run(loss, feed_dict={X: batch_xs, Y: batch_ys})
                avg_cost += cost/total_batch
                if i % 20 == 0:
                    print("Epoch:", '%03d' % (epoch+1), "Step:", '%03d' % i,
                          "Loss:", str(cost))

