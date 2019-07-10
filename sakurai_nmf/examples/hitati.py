"""Main of hitati model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
from agents.tools import AttrDict
import time

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
                   output_debug=False):
    # 1 epoch = num_timesteps * batch_size
    num_timesteps = x_train.shape[0] // batch_size
    for i in range(num_iters):
        # Train...
        start_time = time.time()
        train_losses = []
        test_losses = []

        # Inference and compute loss during training.
        for t in range(num_timesteps):
            x, y = benchmark_model.batch(x_train, y_train, batch_size=batch_size)
            _, train_loss = sess.run([train_op, model.loss], feed_dict={
                model.inputs: x,
                model.labels: y,
            })
            train_losses.append(train_loss)
        duration = time.time() - start_time

        # Compute test accuracy.
        for _ in range(5):
            x, y = benchmark_model.batch(x_test, y_test, batch_size=batch_size)
            test_losses.append(sess.run(model.loss, feed_dict={
                model.inputs: x,
                model.labels: y,
            }))
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)

        print('\r({}/{}) [Train]loss {:.3f}, time, {:.3f} [Test]loss {:.3f}'.format(
            i + 1, num_iters,
            train_loss, duration, test_loss), end='', flush=True)
    print()


def main(_):
    print('use_bias', FLAGS.use_bias)
    # Set configuration
    config = AttrDict(default_config())
    # Build one hot mnist model.
    model = benchmark_model.build_tf_hitachi_model(config.batch_size,
                                                   use_bias=config.use_bias,
                                                   activation=config.activation)

    # Load hitachi data.
    (x_train, y_train), (x_test, y_test) = benchmark_model.load_hitachi_data(config.path, test_size=0.1)

    # Testing whether the dataset have correct shape.
    assert x_train.shape[1] == 3
    assert y_train.shape[1] == 4

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
        _train_and_test(bp_train_op, num_iters=config.num_bp_iters)

        print('NMF-optimizer')
        # Train with NMF optimizer.
        _train_and_test(train_op, num_iters=config.num_mf_iters)



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
