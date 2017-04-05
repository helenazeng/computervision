import tensorflow as tf
import numpy as np
import pylab
import mahotas as mh
import matplotlib.pyplot as plt
color = []
mask = []
normal = []
for i in range(0,1):
    color.append(mh.imread('project/eecs442challenge/train/color/'+str(i)+'.png'))
    mask.append(mh.imread('project/eecs442challenge/train/mask/'+str(i)+'.png'))
    normal.append(mh.imread('project/eecs442challenge/train/normal/'+str(i)+'.png'))


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import l2_normalize
X = color
Y = normal
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
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=2)
# tensorboard_verbose from 0-3 3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.
# check point default = none
model.fit(X, Y, n_epoch=300, shuffle=True,
          show_metric=True, batch_size=1, snapshot_step=500,
          snapshot_epoch=False, run_id='project_test')
#plt.imshow(mh.as_rgb(red, green, blue))
model.save("test2.model")

output = model.predict(X)

# Save the prediction off
val = np.array(output[0])
mhval = mh.as_rgb(val[:,:,0], val[:,:,1], val[:,:,2])
mh.imsave("test2.png", mhval)

plt.imshow(val)
plt.show()


#plt.imshow(mh.as_rgb(output[:,:,0], output[:,:,1], output[:,:,2]))
