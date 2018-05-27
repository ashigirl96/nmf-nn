"""64bit model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import agents
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from sakurai_nmf.losses import frobenius_norm

batch_size = 3000
label_size = 1

def build_tf_model():
    batch_size = 3000
    label_size = 1
    inputs = tf.placeholder(tf.float64, (batch_size, 784), name='inputs')
    labels = tf.placeholder(tf.float64, (batch_size, label_size), name='labels')
    x = tf.layers.dense(inputs, 100, activation=tf.nn.relu, use_bias=True)
    x = tf.layers.dense(x, 50, use_bias=False, activation=tf.nn.relu)
    outputs = tf.layers.dense(x, label_size, activation=None, use_bias=True)
    losses = frobenius_norm(labels, outputs)
    loss = tf.reduce_mean(losses)
    correct_prediction = tf.equal(tf.cast(labels, tf.int32), tf.cast(outputs, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100.
    tf.summary.scalar('accuracy', accuracy)
    
    return agents.tools.AttrDict(inputs=inputs,
                                 outputs=outputs,
                                 labels=labels,
                                 loss=loss,
                                 accuracy=accuracy,
                                 )


def build_tf_cross_entropy_model(batch_size, shape=784, use_bias=False, activation=None, use_softmax=False):
    inputs = tf.placeholder(tf.float64, (batch_size, shape), name='inputs')
    labels = tf.placeholder(tf.float64, (batch_size, 10), name='labels')
    
    x = tf.layers.dense(inputs, 1000, activation=activation, use_bias=use_bias)
    x = tf.layers.dense(x, 500, activation=activation, use_bias=use_bias)
    outputs = tf.layers.dense(x, 10, activation=None, use_bias=use_bias)
    
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=outputs)
    cross_entropy = tf.reduce_mean(losses)
    
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(outputs, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100.
    
    return agents.tools.AttrDict(inputs=inputs,
                                 outputs=outputs,
                                 labels=labels,
                                 cross_entropy=cross_entropy,
                                 accuracy=accuracy,
                                 )

def build_tf_one_hot_model(batch_size, shape=784, use_bias=False, activation=None, use_softmax=False):
    inputs = tf.placeholder(tf.float64, (batch_size, shape), name='inputs')
    labels = tf.placeholder(tf.float64, (batch_size, 10), name='labels')
    
    activation = None or activation
    x = tf.layers.dense(inputs, 1000, activation=activation, use_bias=use_bias)
    x = tf.layers.dense(x, 500, activation=activation, use_bias=use_bias)
    outputs = tf.layers.dense(x, 10, activation=None, use_bias=use_bias)
    
    losses = frobenius_norm(labels, outputs)
    frob_norm = tf.reduce_mean(losses)
    other_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=outputs)
    cross_entropy = tf.reduce_mean(other_losses)
    
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(outputs, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100.
    
    return agents.tools.AttrDict(inputs=inputs,
                                 outputs=outputs,
                                 labels=labels,
                                 frob_norm=frob_norm,
                                 cross_entropy=cross_entropy,
                                 accuracy=accuracy,
                                 )


def build_keras_model():
    batch_size = 3000
    label_size = 1
    inputs = tf.keras.Input((784,), batch_size=batch_size, dtype=tf.float64, name='inputs')
    labels = tf.keras.Input((label_size,), batch_size=batch_size, name='labels', dtype=tf.float64)
    # x = tf.keras.layers.Dense(100, activation=tf.nn.relu, dtype=tf.float64)(inputs)
    # x = tf.keras.layers.Dense(50, activation=tf.nn.relu, use_bias=False, dtype=tf.float64)(x)
    outputs = tf.keras.layers.Dense(label_size, dtype=tf.float64)(inputs)
    losses = tf.keras.losses.mean_squared_error(y_true=labels, y_pred=outputs)
    loss = tf.reduce_mean(losses)
    return inputs, labels, loss


def build_data(batch_size, label_size):
    x = np.random.uniform(0., 1., size=(batch_size, 784)).astype(np.float64)
    y = np.random.uniform(-1., 1., size=(batch_size, label_size)).astype(np.float64)
    return x, y


def load_one_hot_data(dataset='mnist'):
    from keras.datasets.mnist import load_data
    shape = 784
    if dataset == 'fashion':
        from keras.datasets.fashion_mnist import load_data
    elif dataset == 'cifar10':
        from keras.datasets.cifar10 import load_data
        shape = 32 * 32 * 3
    (x_train, y_train), (x_test, y_test) = load_data()
    
    x_train = x_train.reshape((-1, shape)).astype(np.float64) / 255.
    y_train = to_categorical(y_train, 10).astype(np.float64)
    x_test = x_test.reshape((-1, shape)).astype(np.float64) / 255.
    y_test = to_categorical(y_test, 10).astype(np.float64)
    return (x_train, y_train), (x_test, y_test)


def batch(x, y, batch_size):
    rand_index = np.random.choice(len(x), batch_size)
    return x[rand_index], y[rand_index]


def get_train_ops(graph: tf.Graph):
    return graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


def default_config():
    num_epochs = 100
    learning_rate = 0.01
    
    return locals()


def main(_):
    from keras.datasets.mnist import load_data
    (x_train, y_train), (x_test, y_test) = load_data('/tmp/mnist')
    x_train = x_train.reshape((-1, 784)).astype(np.float64) / 255.
    y_train = y_train[..., None].astype(np.float64)
    
    model = build_tf_model()
    config = agents.tools.AttrDict(default_config())
    optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
    train_op = optimizer.minimize(model.loss)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for _ in range(10_000):
            x, y = batch(x_train, y_train, batch_size)
            _, loss, acc = sess.run([train_op, model.loss, model.accuracy], feed_dict={
                model.inputs: x, model.labels: y,
            })
            print('loss {}, acc {}'.format(loss, acc))


def one_hot_main():
    from keras.utils.np_utils import to_categorical
    (x_train, y_train), (x_test, y_test) = load_data('/tmp/mnist')
    x_train = x_train.reshape((-1, 784)).astype(np.float64) / 255.
    y_train = to_categorical(y_train, 10).astype(np.float64)
    
    model = build_tf_one_hot_model()
    config = agents.tools.AttrDict(default_config())
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    train_op = optimizer.minimize(model.loss)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for _ in range(20 + 4):
            x, y = batch(x_train, y_train, batch_size)
            _, loss, acc = sess.run([train_op, model.other_loss, model.accuracy], feed_dict={
                model.inputs: x, model.labels: y,
            })
            print('loss {}, acc {}'.format(loss, acc))


if __name__ == '__main__':
    one_hot_main()