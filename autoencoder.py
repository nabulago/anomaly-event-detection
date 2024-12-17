""" Auto Encoder Example.
Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os,os.path
# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
pathToDataset ="path_to_data_folder/data"
dataSetList = ['motionfeatures.p']
ptdt = 'path_to_data_folder/data'
print(ptdt)
print(dataSetList)
print( pathToDataset+dataSetList[0])
if os.path.isfile(str(pathToDataset)+str(dataSetList[0])): 
	print (str(pathToDataset)+str(dataSetList[0]))
mydataset = open(str(pathToDataset)+str(dataSetList[0]),'rb')
lds= pickle.load(mydataset)
print (lds)
print (lds.shape)
# train, test = train_test_split()
print (np.round(len(lds[1])*0.3).astype('uint8'))
print (np.round(len(lds[1])*0.8).astype('uint8')-np.round(len(lds[1])*0.5).astype('uint8'))
train = lds[:,0:35]
test = lds[:,35:50]
print(train.shape)
print(test.shape)
print(test)
# Training Parameters
learning_rate = 0.01
num_steps = 3000
#batch_size = 256
batch_size = 1
display_step = 1000
examples_to_show = 10

# # Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
#num_hidden_1 = 1024 # 1st layer num features
#num_hidden_2 = 512 # 2nd layer num features (the latent dim)
#num_hidden_3 = 256
#num_hidden_4 = 128
num_input = 225 # MNIST data input (img shape: 15*15)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    # 'encoder_h3': tf.Variable(tf.random_normal([num_hideen_2, num_hidden_3])),
    # 'encoder_h4': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_4])),
    
    # 'decoder_h4': tf.Variable(tf.random_normal([num_hidden_4, num_hidden_3])),
    # 'decoder_h3': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    # 'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
    # 'encoder_b4': tf.Variable(tf.random_normal([num_hidden_4])),
    
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))

    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
print(y_pred)
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        #batch_x, _ = mnist.train.next_batch																													(batch_size)
        batch_x = train.T
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((15 * n, 15 * n))
    canvas_recon = np.empty((15 * n, 15 * n))
    for i in range(n):
        # MNIST test set
        #batch_x, _ = mnist.test.next_batch(n)
        batch_x = test.T
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 15:(i + 1) * 15, j * 15:(j + 1) * 15] = \
                batch_x[j].reshape([15, 15])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 15:(i + 1) * 15, j * 15:(j + 1) * 15] = \
                g[j].reshape([15, 15])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
