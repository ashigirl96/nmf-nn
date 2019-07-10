"""Main of hitati model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
from agents.tools import AttrDict
import time
import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns

# sns.set(color_codes=True)
sns.set(style='whitegrid')
# matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'

from sakurai_nmf import benchmark_model
from sakurai_nmf.optimizer import NMFOptimizer


def default_config():
    # Batch size
    batch_size = FLAGS.batch_size
    # Dataset
    path = FLAGS.path
    # Number of matrix factorization iterations
    num_mf_iters = FLAGS.num_mf_iters
    # Number of back propagation iterations
    num_bp_iters = FLAGS.num_bp_iters
    # Learning rate for adam
    learning_rate = FLAGS.lr
    # NMF actiovation
    if FLAGS.use_relu:
        activation = tf.nn.relu
    else:
        activation = None
    # NMF use bias
    use_bias = FLAGS.use_bias
    return locals()


def train_and_test(train_op, num_iters, sess, model, x_train, y_train, x_test, y_test, batch_size=1,
                   output_debug=False, add_loss_before_train=False) -> Tuple[np.ndarray, np.ndarray]:
    # 1 epoch = num_timesteps * batch_size
    total_train_losses = []
    total_test_losses = []
    num_timesteps = x_train.shape[0] // batch_size

    # Before training
    if add_loss_before_train:
        train_losses = []
        for _ in range(5):
            x, y = benchmark_model.batch(x_train, y_train, batch_size=batch_size)
            train_losses.append(sess.run(model.mse_losses, feed_dict={
                model.inputs: x,
                model.labels: y,
            }))
        total_train_losses.append(np.mean(train_losses, axis=0))
        test_losses = []
        for _ in range(5):
            x, y = benchmark_model.batch(x_test, y_test, batch_size=batch_size)
            test_losses.append(sess.run(model.mse_losses, feed_dict={
                model.inputs: x,
                model.labels: y,
            }))
        total_test_losses.append(np.mean(test_losses))

    durations = [.0]
    for i in range(num_iters):
        # Train...
        start_time = time.time()
        train_losses = []
        test_losses = []

        # Inference and compute loss during training.
        for t in range(num_timesteps):
            x, y = benchmark_model.batch(x_train, y_train, batch_size=batch_size)
            _, train_loss = sess.run([train_op, model.mse_losses], feed_dict={
                model.inputs: x,
                model.labels: y,
            })
            assert train_loss.shape == (4,), "train_loss shape {}".format(train_loss.shape)
            train_losses.append(train_loss)
        duration = time.time() - start_time
        durations.append(duration)

        # Compute test accuracy.
        for _ in range(5):
            x, y = benchmark_model.batch(x_test, y_test, batch_size=batch_size)
            test_losses.append(sess.run(model.mse, feed_dict={
                model.inputs: x,
                model.labels: y,
            }))
        train_loss = np.mean(train_losses, axis=0)
        assert train_loss.shape[0] == 4
        test_loss = np.mean(test_losses)

        print('\r({}/{}) [Train]loss {:.3f}, time, {:.3f} [Test]loss {:.3f}'.format(
            i + 1, num_iters,
            np.mean(train_loss), duration, test_loss), end='', flush=True)
        total_train_losses.append(train_loss)
        total_test_losses.append(test_loss)
    print()
    print('Mean training time {}'.format(np.mean(durations)))

    total_train_losses = np.asarray(total_train_losses).T
    total_test_losses = np.asarray(total_test_losses).T

    return (total_train_losses, total_test_losses)


def main(_):
    print('use_bias', FLAGS.use_bias)
    LABEL_SIZE = 4
    # Set configuration
    config = AttrDict(default_config())
    # Build one hot mnist model.
    model = benchmark_model.build_tf_hitachi_simple_model(config.batch_size,
                                                          use_bias=config.use_bias,
                                                          activation=config.activation,
                                                          label_size=LABEL_SIZE)

    # Load hitachi data.
    (x_train, y_train), (x_test, y_test) = benchmark_model.load_hitachi_data(
        config.path, test_size=0.1)

    # Testing whether the dataset have correct shape.
    # assert x_train.shape[1] == 3
    # assert y_train.shape[1] == 4

    # Minimize model's loss with NMF optimizer.
    # optimizer = NMFOptimizer(config)
    optimizer = NMFOptimizer()
    train_op = optimizer.minimize(model.loss)

    # Minimize model's loss with Adam optimizer.
    bp_optimizer = tf.train.AdamOptimizer(config.learning_rate)
    bp_train_op = bp_optimizer.minimize(model.loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        _train_and_test = functools.partial(train_and_test,
                                            sess=sess, model=model,
                                            x_train=x_train, y_train=y_train,
                                            x_test=x_test, y_test=y_test,
                                            batch_size=config.batch_size)

        print('Adam-optimizer')
        # Train with Adam optimizer.
        bp_train_losses, bp_test_losses = _train_and_test(bp_train_op, num_iters=config.num_bp_iters,
                                                          add_loss_before_train=True)

        print('NMF-optimizer')
        # Train with NMF optimizer.
        mf_train_losses, mf_test_losses = _train_and_test(train_op, num_iters=config.num_mf_iters)

    length_bp = bp_train_losses.shape[1]

    # mf_train_losses.insert(0, bp_train_losses[-1])
    mf_train_losses = np.hstack([bp_train_losses[:, -1, np.newaxis], mf_train_losses])

    # mf_test_losses.insert(0, bp_test_losses[-1])
    length_mf = mf_train_losses.shape[1]

    bp_x = np.arange(0, length_bp)
    mf_x = np.arange(length_bp - 1, length_bp + length_mf - 1)

    plt.rc('grid', linestyle="--", color='black')
    # 'dodgerblue', 'black'

    fig, ax = plt.subplots(2, 2, figsize=(6 * 2.5, 4 * 2.5))

    if config.num_bp_iters > 0:
        ax[0, 0].plot(bp_x, bp_train_losses[0], linestyle='-', color='dodgerblue', label='BP-B{}'.format(1))
        ax[0, 1].plot(bp_x, bp_train_losses[1], linestyle='-', color='dodgerblue', label='BP-B{}'.format(2))
        ax[1, 0].plot(bp_x, bp_train_losses[2], linestyle='-', color='dodgerblue', label='BP-B{}'.format(3))
        ax[1, 1].plot(bp_x, bp_train_losses[3], linestyle='-', color='dodgerblue', label='BP-B{}'.format(4))
    if config.num_mf_iters > 0:
        ax[0, 0].plot(mf_x, mf_train_losses[0], linestyle='-', color='black', label='NMF-B{}'.format(1))
        ax[0, 1].plot(mf_x, mf_train_losses[1], linestyle='-', color='black', label='NMF-B{}'.format(2))
        ax[1, 0].plot(mf_x, mf_train_losses[2], linestyle='-', color='black', label='NMF-B{}'.format(3))
        ax[1, 1].plot(mf_x, mf_train_losses[3], linestyle='-', color='black', label='NMF-B{}'.format(4))
    ax[0, 0].set_title("B1")
    ax[0, 1].set_title("B2")
    ax[1, 0].set_title("B3")
    ax[1, 1].set_title("B4")

    # plt.plot(bp_x, bp_test_losses, linestyle='--', color='dodgerblue', label='test-bp')
    # plt.plot(mf_x, mf_test_losses, linestyle='--', color='black', label='test-nmf')
    plt.xlabel('#Epoch')
    plt.ylabel('')
    # plt.legend()
    plt.grid(True)
    plt.show()

    # print('bp_train', bp_train_losses)
    # print('bp_test', bp_test_losses)
    # print('mf_train', mf_train_losses)


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_integer('batch_size', 300, """Size of batches""")
    tf.flags.DEFINE_string('path', '/tmp', '''dataset path of hitachi''')
    tf.flags.DEFINE_integer('num_mf_iters', 5, '''Number of matrix factorization iterations''')
    tf.flags.DEFINE_integer('num_bp_iters', 25, '''Number of back propagation(adam) iterations''')
    tf.flags.DEFINE_float('lr', 0.001, '''learning rate for back propagation''')
    tf.flags.DEFINE_boolean('use_relu', False, '''Use ReLU''')
    tf.flags.DEFINE_boolean('use_bias', False, '''Use bias''')
    tf.app.run()
