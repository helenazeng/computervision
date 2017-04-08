import tensorflow as tf
import numpy as np
import pylab
import mahotas as mh
import matplotlib.pyplot as plt

color = []
mask = []
normal = []
test = []
for j in range(0,2000):
    test.append(mh.imread('EECS 442/project/eecs442challenge/test/color/'+str(j)+'.png'))

for i in range(0,20000):
    color.append(mh.imread('EECS 442/project/eecs442challenge/train/color/'+str(i)+'.png'))
 #   mask.append(mh.imread('eecs442challenge/train/mask/'+str(i)+'.png'))
    normal.append(mh.imread('EECS 442/project/eecs442challenge/train/normal/'+str(i)+'.png'))


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import l2_normalize

X = color
Y = normal
network = input_data(shape=[None, 128, 128, 3])

network = conv_2d(network, 128, 5, activation='relu') # size 128*128*32
network = conv_2d(network, 512, 5, activation='relu') # size 128*128*64




#network = tflearn.conv_2d_transpose(network,3,5,[128, 128],activation = 'relu')
network = conv_2d(network,3,3,activation = 'relu')
#network = tf.image.resize_images(network,[128,128],tf.image.ResizeMethod.BILINEAR)




#network = conv_2d(network, 192, 3, activation='relu')
#network = conv_2d(network, 128, 3, activation='relu')
#network = max_pool_2d(network, 3, strides=3)

#network = conv_2d(network, 384, 3, activation='relu')
#network = conv_2d(network, 256, 3, activation='relu')
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

#network = fully_connected(network, 512, activation='relu')
#network = dropout(network, 0.5)
##network = fully_connected(network, 4096, activation='relu')
#network = dropout(network, 0.5)
#network = fully_connected(network, 128*128*3, activation='softmax')
#network = tflearn.reshape(network, [-1, 128, 128, 3])
#network = tf.nn.l2_normalize(network,3)

network = regression(network, optimizer='momentum',
                     loss='mean_square',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='none',
                    max_checkpoints=0, tensorboard_verbose=0)
# tensorboard_verbose from 0-3 3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.
# check pont default = none
model.fit(X, Y, n_epoch=20000, shuffle=True,
          show_metric=True, batch_size=250, snapshot_step=500,
          snapshot_epoch=False, run_id='project_test')
#plt.imshow(mh.as_rgb(red, green, blue))

model.save("project.model")

output = model.predict(test)

# Save the prediction off
for j in range(0,2000):
    val = np.array(output[j])
    mhval = mh.as_rgb(val[:,:,0], val[:,:,1], val[:,:,2])
    mh.imsave(str(j)+".png", mhval)

