"""not implement yet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import agents
import collections
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib.rnn import GRUBlockCellV2

from sakurai_nmf.losses import frobenius_norm
from sakurai_nmf.optimizer import utility

Dense = keras.layers.Dense
Input = keras.layers.Input
SimpleRNN = keras.layers.SimpleRNN
Model = keras.Model
kutils = keras.utils
kdatasets = keras.datasets


def search(x):
    queue = collections.deque([x])
    variable_names = []
    explored_inputs = {x}
    
    # We do a BFS on the dependency graph of the input function to find
    # the variables.
    while len(queue) != 0:
        tf_obj = queue.popleft()
        print(tf_obj)
        # print(tf_obj)
        if tf_obj is None:
            continue
        # The object put into the queue is not necessarily an operation,
        # so we want the op attribute to get the operation underlying the
        # object. Only operations contain the inputs that we can explore.
        if hasattr(tf_obj, "op"):
            tf_obj = tf_obj.op
        for input_op in tf_obj.inputs:
            if input_op not in explored_inputs:
                queue.append(input_op)
                explored_inputs.add(input_op)
        for control in tf_obj.control_inputs:
            if control not in explored_inputs:
                queue.append(control)
                explored_inputs.add(control)


def build_rnn_mnist(batch_size, use_bias=False, activation=None):
    time_steps = 28
    num_features = 28
    inputs = tf.keras.layers.Input((time_steps, num_features), batch_size=batch_size, dtype=tf.float64, name='inputs')
    labels = tf.placeholder(tf.float64, (batch_size, 10), name='labels')
    
    activation = None or activation
    x = tf.keras.layers.SimpleRNN(100, use_bias=use_bias, activation=activation, return_sequences=True)(inputs)
    x = x[:, -1, :]
    # x = tf.keras.layers.SimpleRNN(100, use_bias=use_bias, activation=activation)(inputs)
    # print(x)
    # print(search(x))
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


def build_tf_rnn_mnist(batch_size, use_bias=False, activation=None):
    # sequence_length = tf.32  # Timesteps
    # length = tf.placeholder(tf.int32, None)
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    max_length = 100
    frame_size = 64
    num_hidden = 200

    sequence = tf.placeholder(
        tf.float32, [None, max_length, frame_size])
    output, state = tf.nn.dynamic_rnn(
        tf.contrib.rnn.GRUCell(num_hidden),
        sequence,
        dtype=tf.float32,
        sequence_length=length(sequence),
    )
    print(length(sequence))


def test_sequence_length():
    # Create input data
    X = np.random.randn(2, 10, 8)

    # The second example is of length 6
    X_lengths = [100, 60]

    cell = tf.nn.rnn_cell.GRUCell(num_units=64)
    print(cell)

    outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float64,
        # sequence_length=X_lengths,
        inputs=X)
    print(outputs)
    print(last_states)



def build_keras_rnn_mnist(batch_size, use_bias=False, activation=None):
    time_steps = 28
    num_features = 28
    inputs = Input(batch_shape=(batch_size, time_steps, num_features), dtype=tf.float64, name='inputs')
    
    activation = None or activation
    x = SimpleRNN(100, use_bias=use_bias, activation=activation)(inputs)
    outputs = Dense(10, activation=None, use_bias=use_bias)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def load_mnist(dataset='mnist'):
    load_data = kdatasets.mnist.load_data
    if dataset == 'fashion':
        load_data = kdatasets.fashion_mnist.load_data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    x_train = x_train.astype(np.float64) / 255.
    y_train = kutils.to_categorical(y_train, 10).astype(np.float64)
    x_test = x_test.astype(np.float64) / 255.
    y_test = kutils.to_categorical(y_test, 10).astype(np.float64)
    return (x_train, y_train), (x_test, y_test)


def batch(x, y, batch_size):
    rand_index = np.random.choice(len(x), batch_size)
    return x[rand_index], y[rand_index]


def _train():
    batch_size = 150
    epoch_size = 3
    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_rnn_mnist(batch_size=batch_size, use_bias=False, activation=tf.nn.relu)
    
    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = optimizer.minimize(model.cross_entropy)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for epoch in range(epoch_size):
            for _ in range(len(x_train) // batch_size):
                # for _ in range(1):
                x, y = batch(x_train, y_train, batch_size)
                _, = sess.run([train_op],
                              feed_dict={model.inputs: x, model.labels: y})
            x, y = batch(x_test, y_test, batch_size)
            loss, acc = sess.run([model.cross_entropy, model.accuracy],
                                 feed_dict={model.inputs: x, model.labels: y})
            print('({}/{}) loss {}, accuracy {}'.format(epoch + 1, epoch_size, loss, acc))


def _vanilla_rnn():
    batch_size = 100
    epoch_size = 1
    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_rnn_mnist(batch_size=batch_size, use_bias=True, activation=tf.nn.relu)
    ops = utility.get_train_ops()
    layers = utility._zip_layer(model.inputs, model.frob_norm, ops)
    variables = utility.TensorFlowVariables(model.frob_norm)


def _keras_rnn():
    batch_size = 100
    model = build_keras_rnn_mnist(batch_size=batch_size)
    print(dir(model.layers[1]))
    print(model.layers[1].trainable_weights)
    print(model.layers[1].trainable)
    print(model.layers[1].units)
    print(type(model.layers[1]))


def cell_unit_rnn():
    RNN = keras.layers.RNN
    # import keras.backend.tensorflow_backend as K
    
    class MinimalRNNCell(keras.layers.Layer):
        def __init__(self, units, **kwargs):
            self.units = units
            self.state_size = units
            super(MinimalRNNCell, self).__init__(**kwargs)
        
        def build(self, input_shape):
            self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                          initializer='uniform',
                                          name='kernel')
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='uniform',
                name='recurrent_kernel')
            self.built = True
        
        def call(self, inputs, states):
            prev_output = states[0]
            h = K.dot(inputs, self.kernel)
            print('hidden', h)
            output = h + K.dot(prev_output, self.recurrent_kernel)
            return output, [output]
    
    # Let's use this cell in a RNN layer:
    
    cell = MinimalRNNCell(100)
    x = keras.Input(batch_shape=(3000, 28, 28))
    # When return_state is True,
    # [<tf.Tensor 'rnn_1/TensorArrayReadV3:0' shape=(3000, 100) dtype=float64>,
    # <tf.Tensor 'rnn_1/while/Exit_2:0' shape=(3000, 100) dtype=float64>]
    
    # When return_state is False,
    # Tensor("rnn_1/TensorArrayReadV3:0", shape=(3000, 100), dtype=float64)
    layer = RNN(cell, return_state=True)
    y = layer(x)
    # model = keras.Model(inputs=x, outputs=y)
    # print(y)
    # variables = utility.TensorFlowVariables(y)
    # model.summary()


def main(_):
    # cell_unit_rnn()
    # _train()
    # build_tf_rnn_mnist(100)
    test_sequence_length()


if __name__ == '__main__':
    tf.app.run()