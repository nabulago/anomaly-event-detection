import tensorflow as tf
import numpy as np
import math
from utils import corrupt
# from libs.utils import corrupt
import pickle
import sys

listOfDatasets = ['apperance_features_train.p','apperance_features_test.p','motion_features_train.p','motion_features_original_test.p']

# Provide data-set path here
datasetPath = 'apperancefeatures.p'
#datasetPath = 'apperancedataset.p'

opendataset = open(datasetPath,'r')
dataset = pickle.load(opendataset)
opendataset.close()

# opendataset = open(datasetPath,'r')
# dataset = pickle.load(opendataset)
# opendataset.close()



if sys.version_info.major == 3:
    print dataset[:, 0:500].shape
    print dataset[:, 501:700].shape
else:
    print (dataset[:, 0:500].shape)
    print (dataset[:, 501:700].shape)


# %%
#def autoencoder(dimensions=[784, 512, 256, 64]):
def autoencoder(dimensions=[225, 1024, 512, 256, 64]):

    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # input to the network
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    print tf.shape(x)
    # Probability that we will corrupt input.
    # This is the essence of the denoising autoencoder, and is pretty
    # basic.  We'll feed forward a noisy input, allowing our network
    # to generalize better, possibly, to occlusions of what we're
    # really interested in.  But to measure accuracy, we'll still
    # enforce a training signal which measures the original image's
    # reconstruction cost.
    #
    # We'll change this to 1 during training
    # but when we're ready for testing/production ready environments,
    # we'll put it back to 0.
    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

    # Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        print "Layer : " + str(layer_i)
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        # output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
        current_input = output
        print output
    # latent representation
    z = current_input
    # Here use the classifier for the latent representaion
    
    encoder.reverse()
    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        # output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
        current_input = output
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    print x.shape
    print tf.shape(x)
    print y.shape
    print tf.shape(y)

    # cost = - tf.add(tf.matmul(tf.transpose(x), tf.log(y)), tf.matmul(tf.transpose(1-x), tf.log(1-y)))
    cost = - tf.add(tf.matmul(x,tf.transpose(tf.log(y))), tf.matmul(1-x, tf.transpose( tf.log(1-y) ) ))
    # cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
    return {'x': x, 'z': z, 'y': y,
            'corrupt_prob': corrupt_prob,
            'cost': cost}

# %%


def test_mnist():
    import tensorflow as tf
    import matplotlib.pyplot as plt

    # %%
    # load Dataset

    mnist = dataset # Here we will set out dataset
    mean_img = np.mean(mnist)
    mnist_train, mnist_test = dataset[:,0:35], dataset[:,36:51]
    print "Train slice of dataset" + str(mnist_train.shape)
    print "Test slice of dataset" + str(mnist_test.shape)
    mean_img = np.mean(mnist_train, axis=1)
    print "Mean Image : "+str(mean_img.shape)
    ae = autoencoder(dimensions=[225, 1024, 512, 256, 64])

    # %%
    learning_rate = 0.001
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    batch_size = 2
    # batch_size = 50
    n_epochs = 10
    for epoch_i in range(n_epochs):
        # print mnist_train.shape[1] // batch_size
        for batch_i in range(mnist_train.shape[1] // batch_size):
        # for batch_i in range(mnist.train.num_examples // batch_size):
        #     batch_xs, _ = mnist.train.next_batch(batch_size)
            print batch_i
            batch_xs = mnist_train[:,batch_i:batch_i + batch_size]
            print "Batch_Xs shape "+str(batch_xs.shape)
            print "Mean Image " + str(mean_img.shape)


            for img in batch_xs.T:
                print "Image shape : " + str(img.shape)
                print "Mean Shape: " + str(mean_img.T.shape)

            train = np.array([img.T - mean_img for img in batch_xs.T])
            sess.run(optimizer, feed_dict={
                ae['x']: train, ae['corrupt_prob']: [1.0]})
        print(epoch_i, sess.run(ae['cost'], feed_dict={
            ae['x']: train, ae['corrupt_prob']: [1.0]}))

    # %%
    # Plot example reconstructions
    n_examples = 15
    # test_xs, _ = mnist.test.next_batch(n_examples)
    # test_xs = mnist_test.T[batch_i:batch_i + batch_size, :]
    test_xs = mnist_test.T
    print "Testxs : " +str(test_xs.shape)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    print "Test xs Norm : " + str(test_xs_norm.shape)

    recon = sess.run(ae['y'], feed_dict={
        ae['x']: test_xs_norm, ae['corrupt_prob']: [0.0]})
    print "Reconstruction shape: " + str(recon.shape)
    print "Reconstruction Complete"

    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))

    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            # np.reshape(test_xs[example_i, :], (28, 28)))
            np.reshape(test_xs[example_i, :], (15, 15)))
        axs[1][example_i].imshow(
            # np.reshape([recon[example_i, :] + mean_img], (28, 28)))
            np.reshape([recon[example_i, :] + mean_img], (15, 15)))
    print 'Plot complete now showing...'
    fig.show()
    plt.draw()
    plt.title("1st function - mnist ones but used our dataset")
    plt.waitforbuttonpress()


def train_appearance_features():
    import tensorflow as tf
    import matplotlib.pyplot as plt

    # %%
    # load Dataset

    appearance_dataset = dataset # Here we will set out dataset
    mean_img = np.mean(appearance_dataset)
    appearance_train, appearance_test = dataset[:,0:35], dataset[:,36:51]
    print appearance_train.shape
    print appearance_test.shape
    # mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder(dimensions=[225, 1024, 512, 256, 64])

    # %%
    learning_rate = 0.001
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    batch_size = 6
    # batch_size = 50
    n_epochs = 2
    for epoch_i in range(n_epochs):
        # print mnist_train.shape[1] // batch_size
        for batch_i in range(appearance_train.shape[1] // batch_size):
            batch_xs = appearance_train.T[batch_i:batch_i + batch_size,:]
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={
                ae['x']: train, ae['corrupt_prob']: [1.0]})
        print(epoch_i, sess.run(ae['cost'], feed_dict={
            ae['x']: train, ae['corrupt_prob']: [1.0]}))

    # %%
    # Plot example reconstructions
    n_examples = 15
    # test_xs, _ = mnist.test.next_batch(n_examples)
    for batch_i in range(appearance_train.shape[1]//batch_size):
        print batch_i, appearance_train.shape[1],batch_size
        test_xs = appearance_test.T[batch_i:batch_i+batch_size,:]
        test_xs_norm = np.array([img - mean_img for img in test_xs])
        recon = sess.run(ae['y'], feed_dict={
        ae['x']: test_xs_norm, ae['corrupt_prob']: [0.0]})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            # np.reshape(test_xs[example_i, :], (28, 28)))
            np.reshape(test_xs[example_i, :], (15, 15)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :] + mean_img], (15, 15)))
    fig.show()
    plt.draw()
    plt.title('Appearance features')
    plt.waitforbuttonpress()

def train_motion_features():
    pass

def train_joint_features():
    # type: () -> object
    pass

if __name__ == '__main__':
    test_mnist()
    # train_appearance_features()
    # train_motion_features()
    # train_joint_features()