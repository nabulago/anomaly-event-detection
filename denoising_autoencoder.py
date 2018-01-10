import tensorflow as tf
import numpy as np
import math
from utils import corrupt
# from libs.utils import corrupt
import pickle

lostOfDatasets = ['motionfeatures.p','motionfeaturesoriginal.p']
datasetPath = 'motionfeatures.p'
#datasetPath = 'apperancedataset.p'

opendataset = open(datasetPath,'r')
dataset = pickle.load(opendataset)
opendataset.close()

# opendataset = open(datasetPath,'r')
# dataset = pickle.load(opendataset)
# opendataset.close()

print dataset[:,0:500].shape
print dataset[:,501:700].shape


# for i,loD in enumerate(lostOfDatasets):
#     # This will load all the datasets and split the dataset
#     #pickle_file = 'notMNIST.pickle'
#     pickle_file = loD

#     with open(pickle_file, 'rb') as f:
#       save = pickle.load(f)
#       train_dataset = save['train_dataset']
#       train_labels = save['train_labels']
#       valid_dataset = save['valid_dataset']
#       valid_labels = save['valid_labels']
#       test_dataset = save['test_dataset']
#       test_labels = save['test_labels']
#       del save  # hint to help gc free up memory
#       print 'Training set' + str(train_dataset.shape) + str(train_labels.shape)
#       print 'Validation set' + str(valid_dataset.shape) + str(valid_labels.shape)
#       print 'Test set' + str(test_dataset.shape) + str(test_labels.shape)


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
    # import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    # %%
    # load MNIST as before

    mnist = dataset # Here we will set out dataset
    # mnist1 = dataset1
    mean_img = np.mean(mnist)
    mnist_train, mnist_test = dataset[:,0:35], dataset[:,36:51]
    print mnist_train.shape
    print mnist_test.shape
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder(dimensions=[225, 1024, 512, 256, 64])
    ae1 = autoencoder(dimensions=[225, 1024, 512, 256, 64])
    ae2 = autoencoder(dimensions=[450, 2048, 1024, 512, 256, 64])
    # ae = autoencoder(dimensions=[784, 256, 64])

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
        for batch_i in range(mnist_train.shape[1] // batch_size):
        # for batch_i in range(mnist.train.num_examples // batch_size):
            # batch_xs, _ = mnist.train.next_batch(batch_size)
            # print batch_i
            batch_xs = mnist_train.T[batch_i:batch_i + batch_size,:]
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={
                ae['x']: train, ae['corrupt_prob']: [1.0]})
        print(epoch_i, sess.run(ae['cost'], feed_dict={
            ae['x']: train, ae['corrupt_prob']: [1.0]}))

    # %%
    # Plot example reconstructions
    n_examples = 15
    # test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs = mnist_test.T
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={
        ae['x']: test_xs_norm, ae['corrupt_prob']: [0.0]})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            # np.reshape(test_xs[example_i, :], (28, 28)))
            np.reshape(test_xs[example_i, :], (15, 15)))
        axs[1][example_i].imshow(
            # np.reshape([recon[example_i, :] + mean_img], (28, 28)))
            np.reshape([recon[example_i, :] + mean_img], (15, 15)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()

if __name__ == '__main__':
    test_mnist()
