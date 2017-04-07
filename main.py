import tensorflow as tf
import numpy as np
import mahotas as mh

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import l2_normalize


data_root = '/home/iblaauw/eecs442/project/eecs442challenge'

num_epochs = 5
num_examples = 20000
batch_size = 128

def create_network():
    network = input_data(shape=[None, 128, 128, 3])

    network = conv_2d(network, 3, 5, activation='relu') # 128 x 128 x 3
    print(network.shape)
    #network = max_pool_2d(network, 2, strides=2) # 62 x 62 x 30
    #network = conv_2d(network, 64, 5, activation='relu') #
    #network = tflearn.conv_2d_transpose(network, 3, 5, [128, 128], activation='relu')

    #network = tf.nn.l2_normalize(network,3)

    #network = tflearn.fully_connected(network, 128*128*3, activation='relu')
    #network = tflearn.reshape(network, [-1, 128, 128, 3])

    def cosine_dist(*args, **kwargs):
        return tf.losses.cosine_distance(*args, dim=3, **kwargs)

    network = regression(network, optimizer='momentum',
                         loss=cosine_dist,
                         learning_rate=0.005)

    return network

def train_network(network, X, mask, Y):
    model = tflearn.DNN(network, checkpoint_path='none',
                        max_checkpoints=0, tensorboard_verbose=0)

    tflearn.data_flow.DataFlow (model, num_threads=8, max_queue=32, shuffle=False, continuous=False, ensure_data_order=False, dprep_dict=None, daug_dict=None)
    # tensorboard_verbose from 0-3 3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.
    # check point default = none
    model.fit(X, Y, n_epoch=num_epochs, shuffle=True,
              show_metric=True, batch_size=batch_size, snapshot_step=500,
              snapshot_epoch=False, run_id='project_test')
    return model


def load_data():
    color = []
    mask = []
    normal = []

    for i in range(0, num_examples):
        color.append(mh.imread(data_root+'/train/color/'+str(i)+'.png'))
        mask.append(mh.imread(data_root+'/train/mask/'+str(i)+'.png'))
        normal.append(mh.imread(data_root+'/train/normal/'+str(i)+'.png'))

    return color, mask, normal



if __name__ == "__main__":
    print("ERROR: Please do not run this file, it does not do anything on its own. Please run train.py or evaluate.py instead.")


