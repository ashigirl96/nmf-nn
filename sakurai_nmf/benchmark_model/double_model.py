"""64bit model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import agents
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sakurai_nmf.losses import frobenius_norm

kutils = keras.utils
kdatasets = keras.datasets

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


def build_tf_hitachi_model(batch_size, feature_size=3, label_size=4, use_bias=False, activation=None):
    num_layers = 7
    units = 128
    inputs = tf.placeholder(tf.float64, (batch_size, feature_size), name='inputs')
    labels = tf.placeholder(tf.float64, (batch_size, label_size), name='labels')
    x = inputs
    for i in range(num_layers):
        x = tf.layers.dense(x, units=units, activation=activation, use_bias=use_bias)
    outputs = tf.layers.dense(x, label_size, activation=None, use_bias=use_bias)
    losses = frobenius_norm(labels, outputs)
    loss = tf.reduce_mean(losses)
    return agents.tools.AttrDict(inputs=inputs,
                                 outputs=outputs,
                                 labels=labels,
                                 loss=loss,
                                 )


def build_tf_hitachi_simple_model(batch_size, feature_size=3, label_size=4, use_bias=False, activation=None):
    inputs = tf.placeholder(tf.float64, (batch_size, feature_size), name='inputs')
    labels = tf.placeholder(tf.float64, (batch_size, label_size), name='labels')
    x = tf.layers.dense(inputs, 1000, activation=activation, use_bias=use_bias)
    x = tf.layers.dense(x, 500, activation=activation, use_bias=use_bias)
    outputs = tf.layers.dense(x, label_size, activation=None, use_bias=use_bias)
    losses = frobenius_norm(labels, outputs)
    loss = tf.reduce_mean(losses)
    mse = loss / tf.sqrt(tf.constant(batch_size, tf.float64))  # teached from SAKURAI
    # mse_losses = losses / tf.sqrt(tf.constant(batch_size, tf.float64))  # teached from SAKURAI
    mse_losses = losses
    return agents.tools.AttrDict(inputs=inputs,
                                 outputs=outputs,
                                 labels=labels,
                                 loss=loss,
                                 mse=mse,
                                 mse_losses=mse_losses,
                                 )


def build_tf_cross_entropy_model(batch_size, shape=784, use_bias=False, activation=None, use_softmax=False):
    inputs = tf.placeholder(tf.float64, (batch_size, shape), name='inputs')
    labels = tf.placeholder(tf.float64, (batch_size, 10), name='labels')

    x = tf.layers.dense(inputs, 1000, activation=activation, use_bias=use_bias)
    x = tf.layers.dense(x, 500, activation=activation, use_bias=use_bias)
    outputs = tf.layers.dense(x, 10, activation=None, use_bias=use_bias)
    probs = tf.nn.softmax(outputs)

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=outputs)
    cross_entropy = tf.reduce_mean(losses)

    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(probs, 1))
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


def load_hitachi_data(path, test_size=0.2, objective='ALL'):
    from pathlib import Path
    import pandas as pd
    from sklearn.model_selection import train_test_split

    path = Path(path)
    assert path.stem == 'test_data_20170623_1a', "path must be set as hitach.csv"
    raw_datasets = pd.read_csv(path)
    if objective is 'ALL':
        objective = ['B1', 'B2', 'B3', 'B4']
    if isinstance(objective, str):
        objective = [objective]

    raw_x_train = raw_datasets[['A1', 'A2', 'A3']]
    raw_y_train = raw_datasets[objective]

    x_train, x_test, y_train, y_test = train_test_split(raw_x_train, raw_y_train, test_size=test_size)
    # convert pandas DataFrame to numpy array
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values
    assert x_train.shape[1] == x_test.shape[1] and x_train.shape[0] == y_train.shape[0]

    return (x_train, y_train), (x_test, y_test)


def load_one_hot_data(dataset='mnist'):
    load_data = kdatasets.mnist.load_data
    shape = 784
    if dataset == 'fashion':
        load_data = kdatasets.fashion_mnist.load_data
    elif dataset == 'cifar10':
        load_data = kdatasets.cifar10.load_data
        shape = 32 * 32 * 3
    (x_train, y_train), (x_test, y_test) = load_data()

    x_train = x_train.reshape((-1, shape)).astype(np.float64) / 255.
    y_train = kutils.to_categorical(y_train, 10).astype(np.float)
    x_test = x_test.reshape((-1, shape)).astype(np.float64) / 255.
    y_test = kutils.to_categorical(y_test, 10).astype(np.float64)
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
    load_data = kdatasets.mnist.load_data
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
    load_data = kdatasets.mnist.load_data
    (x_train, y_train), (x_test, y_test) = load_data('/tmp/mnist')
    x_train = x_train.reshape((-1, 784)).astype(np.float64) / 255.
    y_train = kutils.to_categorical(y_train, 10).astype(np.float64)

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
