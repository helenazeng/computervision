import tensorflow as tf
import numpy as np
import mahotas as mh

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import l2_normalize


data_root = '/home/iblaauw/eecs442/project/eecs442challenge'


def create_network():
    network = input_data(shape=[None, 128, 128, 3])

    network = conv_2d(network, 64, 5, activation='relu')
    #network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=3)

    network = conv_2d(network, 192, 3, activation='relu')
    #network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=3)

    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    #network = conv_2d(network, 256, 3, activation='relu')
    #network = max_pool_2d(network, 2, strides=2)

    #network = conv_2d(network, 512, 3, activation='relu')
    #network = conv_2d(network, 512, 3, activation='relu')
    #network = conv_2d(network, 512, 3, activation='relu')
    #network = max_pool_2d(network, 2, strides=2)

    #network = conv_2d(network, 512, 3, activation='relu')
    #network = conv_2d(network, 512, 3, activation='relu')
    #network = conv_2d(network, 512, 3, activation='relu')
    #network = max_pool_2d(network, 2, strides=2)

    network = fully_connected(network, 512, activation='relu')
    #network = dropout(network, 0.5)
    ##network = fully_connected(network, 4096, activation='relu')
    #network = dropout(network, 0.5)
    network = fully_connected(network, 128*128*3, activation='softmax')
    network = tflearn.reshape(network, [-1, 128, 128, 3])
    network = tf.nn.l2_normalize(network,3)

    network = regression(network, optimizer='momentum',
                         loss='mean_square',
                         learning_rate=0.005)

    return network

def train_network(network, X, mask, Y):
    model = tflearn.DNN(network, checkpoint_path='model_vgg',
                        max_checkpoints=1, tensorboard_verbose=2)

    tflearn.data_flow.DataFlow (model, num_threads=8, max_queue=32, shuffle=False, continuous=False, ensure_data_order=False, dprep_dict=None, daug_dict=None)
    # tensorboard_verbose from 0-3 3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.
    # check point default = none
    model.fit(X, Y, n_epoch=10, shuffle=True,
              show_metric=True, batch_size=20, snapshot_step=500,
              snapshot_epoch=False, run_id='project_test')


def load_data():
    color = []
    mask = []
    normal = []

    for i in range(0, 1000):
        color.append(mh.imread(data_root+'/train/color/'+str(i)+'.png'))
        mask.append(mh.imread(data_root+'/train/mask/'+str(i)+'.png'))
        normal.append(mh.imread(data_root+'/train/normal/'+str(i)+'.png'))

    return color, mask, normal



if __name__ == "__main__":
    print("ERROR: Please do not run this file, it does not do anything on its own. Please run train.py or evaluate.py instead.")


