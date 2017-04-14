import tensorflow as tf
import numpy as np
import mahotas as mh

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import l2_normalize


data_root = '.'


def create_network():
    network = input_data(shape=[None, 128, 128, 3])
    
    

    network = conv_2d(network, 32, 5, activation='relu') # size 128*128*32
    network = conv_2d(network, 64, 3, activation='relu') # size 128*128*64

    #network = tflearn.conv_2d_transpose(network,3,5,[128, 128],activation = 'relu')
    network = conv_2d(network,3,3,activation = 'relu')

    def cosine_dist(*args, **kwargs):
        return tf.losses.cosine_distance(*args, dim=3, **kwargs)

    network = regression(network, optimizer='momentum',
                         loss='hinge_loss',
                         learning_rate=0.01)
    
    return network

def train_network(network, X, mask, Y):
    model = tflearn.DNN(network, checkpoint_path='model_5',
                        max_checkpoints=1, tensorboard_verbose=2)

    
    model.fit(X, Y, n_epoch=5, shuffle=True,
              show_metric=True, batch_size=25, snapshot_step=500,
              snapshot_epoch=False, run_id='gigantic')
    return model

def load_data():
    color = []
    mask = []
    normal = []

    for i in range(0, 20000):
        color.append(mh.imread(data_root+'/train/color/'+str(i)+'.png'))
        mask.append(mh.imread(data_root+'/train/mask/'+str(i)+'.png'))
        normal.append(mh.imread(data_root+'/train/normal/'+str(i)+'.png'))

    return color, mask, normal



if __name__ == "__main__":
    print("ERROR: Please do not run this file, it does not do anything on its own. Please run train.py or evaluate.py instead.")


