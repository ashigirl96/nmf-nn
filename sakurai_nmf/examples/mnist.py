"""Main of (fashion) mnist model."""

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
    dataset = FLAGS.dataset
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
    for i in range(num_iters):
        # Train...
        start_time = time.time()
        x, y = benchmark_model.batch(x_train, y_train, batch_size=batch_size)
        _, = sess.run([train_op], feed_dict={
            model.inputs: x,
            model.labels: y,
        })
        duration = time.time() - start_time
        
        train_loss, train_acc = sess.run([model.frob_norm, model.accuracy], feed_dict={
            model.inputs: x,
            model.labels: y,
        })
        if output_debug:
            outputs = sess.run(model.outputs, feed_dict={
                model.inputs: x,
                model.labels: y,
            })
            print(outputs)
        stats = []
        # Compute test accuracy.
        for _ in range(5):
            x, y = benchmark_model.batch(x_test, y_test, batch_size=batch_size)
            stats.append(sess.run([model.frob_norm, model.accuracy], feed_dict={
                model.inputs: x,
                model.labels: y,
            }))
        test_loss, test_acc = np.mean(stats, axis=0)

        print('\r({}/{}) [Train]loss {:.3f}, accuracy {:.3f} time, {:.3f} [Test]loss {:.3f}, accuracy {:.3f}'.format(
            i + 1, num_iters,
            train_loss, train_acc, duration, test_loss, test_acc), end='', flush=True)
    print()


def main(_):
    print('use_bias', FLAGS.use_bias)
    # Set configuration
    config = AttrDict(default_config())
    # Build one hot mnist model.
    model = benchmark_model.build_tf_one_hot_model(batch_size=config.batch_size,
                                                   use_bias=config.use_bias,
                                                   activation=config.activation)
    # Load one hot mnist data.
    (x_train, y_train), (x_test, y_test) = benchmark_model.load_one_hot_data(dataset=config.dataset)
    
    # Testing whether the dataset have correct shape.
    assert x_train.shape == (60000, 784)
    assert y_train.shape == (60000, 10)
    
    # Minimize model's loss with NMF optimizer.
    # optimizer = NMFOptimizer(config)
    optimizer = NMFOptimizer()
    train_op = optimizer.minimize(model.frob_norm)
    
    # Minimize model's loss with Adam optimizer.
    bp_optimizer = tf.train.AdamOptimizer(config.learning_rate)
    bp_train_op = bp_optimizer.minimize(model.cross_entropy)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        _train_and_test = functools.partial(train_and_test,
                                            sess=sess, model=model,
                                            x_train=x_train, y_train=y_train,
                                            x_test=x_test, y_test=y_test,
                                            batch_size=config.batch_size)
        
        print('NMF-optimizer')
        # Train with NMF optimizer.
        _train_and_test(train_op, num_iters=config.num_mf_iters)
        
        print('Adam-optimizer')
        # Train with Adam optimizer.
        _train_and_test(bp_train_op, num_iters=config.num_bp_iters)


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('batch_size', 3000, """Size of batches""")
    tf.app.flags.DEFINE_string('dataset', 'mnist', '''mnist or fashion''')
    tf.app.flags.DEFINE_integer('num_mf_iters', 3, '''Number of matrix factorization iterations''')
    tf.app.flags.DEFINE_integer('num_bp_iters', 5, '''Number of back propagation(adam) iterations''')
    tf.app.flags.DEFINE_float('lr', 0.001, '''learning rate for back propagation''')
    tf.app.flags.DEFINE_boolean('use_relu', False, '''Use ReLU''')
    tf.app.flags.DEFINE_boolean('use_bias', False, '''Use bias''')
    tf.app.run()
