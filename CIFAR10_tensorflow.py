
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[13]:


# Helper functions

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def conv_norm_layer(input, shape, phase):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu( batch_norm_wrapper( conv2d(input, W) + b, phase))

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

# Batch Nomalization 
def batch_norm_wrapper(inputs, is_training, decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training == True:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


# In[12]:


import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# CIFAR10 데이터 경로 
DATA_PATH = "./cifar-10-batches-py"
BATCH_SIZE = 50
STEPS = 500000
epsilon = 1e-3

def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
    return dict


def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()


class CifarLoader(object):
    """
    Load and mange the CIFAR dataset.
    (for any practical use there is no reason not to use the built-in dataset handler instead)
    """
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i+batch_size], self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

    def random_batch(self, batch_size):
        n = len(self.images)
        ix = np.random.choice(n, batch_size)
        return self.images[ix], self.labels[ix]

class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)]).load()
        self.test = CifarLoader(["test_batch"]).load()


def run_simple_net():
    cifar = CifarDataManager()
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    phase = tf.placeholder(tf.bool) 

    conv1 = conv_norm_layer(x, [5, 5, 3, 32], phase)
    conv1_pool = max_pool_2x2(conv1)

    conv2 = conv_norm_layer(conv1_pool, [5, 5, 32, 64], phase)
    conv2_pool = max_pool_2x2(conv2)

    conv3 = conv_norm_layer(conv2_pool, [5, 5, 64, 128], phase)
    conv3_pool = max_pool_2x2(conv3)
    conv3_flat = tf.reshape(conv3_pool, [-1, 4 * 4 * 128])
    conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

    full_1 = tf.nn.relu(full_layer(conv3_drop, 512))
    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    y_conv = full_layer(full1_drop, 10)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test(sess):
        X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
        Y = cifar.test.labels.reshape(10, 1000, 10)
        acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], phase: False, keep_prob: 1.0})
                       for i in range(10)])
        print("Accuracy: {:.4}%".format(acc * 100))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            batch = cifar.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], phase: True,  keep_prob: 0.5})

            if i % 500 == 0:
                test(sess)

        test(sess)


def build_second_net():
    cifar = CifarDataManager()
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    phase = tf.placeholder(tf.bool) 

    C1, C2, C3 = 32, 64, 128
    F1 = 600

    conv1_1 = conv_norm_layer(x, [3, 3, 3, C1], phase)
    conv1_2 = conv_norm_layer(conv1_1, [3, 3, C1, C1], phase)
    conv1_3 = conv_norm_layer(conv1_2, [3, 3, C1, C1], phase)
    conv1_pool = max_pool_2x2(conv1_3)
    conv1_drop = tf.nn.dropout(conv1_pool, keep_prob=keep_prob)

    conv2_1 = conv_norm_layer(conv1_drop, [3, 3, C1, C2], phase)
    conv2_2 = conv_norm_layer(conv2_1, [3, 3, C2, C2], phase)
    conv2_3 = conv_norm_layer(conv2_2, [3, 3, C2, C2], phase)
    conv2_pool = max_pool_2x2(conv2_3)
    conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)

    conv3_1 = conv_norm_layer(conv2_drop, [3, 3, C2, C3], phase)
    conv3_2 = conv_norm_layer(conv3_1, [3, 3, C3, C3], phase)
    conv3_3 = conv_norm_layer(conv3_2, [3, 3, C3, C3], phase)
    conv3_pool = tf.nn.max_pool(conv3_3, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    conv3_flat = tf.reshape(conv3_pool, [-1, C3])
    conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

    full1 = tf.nn.relu(full_layer(conv3_flat, F1))
    full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

    y_conv = full_layer(full1_drop, 10)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test(sess):
        X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
        Y = cifar.test.labels.reshape(10, 1000, 10)
        acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], phase: False , keep_prob: 1.0})
                       for i in range(10)])
        print("Accuracy: {:.4}%".format(acc * 100))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            batch = cifar.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], phase: True, keep_prob: 0.5})

            if i % 500 == 0:
                test(sess)

        test(sess)


def create_cifar_image():
    d = CifarDataManager()
    print("Number of train images: {}".format(len(d.train.images)))
    print("Number of train labels: {}".format(len(d.train.labels)))
    print("Number of test images: {}".format(len(d.test.images)))
    print("Number of test images: {}".format(len(d.test.labels)))
    images = d.train.images
    display_cifar(images, 10)


# In[5]:


create_cifar_image()


# In[14]:


run_simple_net()


# In[15]:


build_second_net()

