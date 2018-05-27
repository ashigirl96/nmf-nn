"""Optimize neural network with Non-negative Matrix Factorization"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sakurai_nmf.matrix_factorization as mf
from . import utility


class NMFOptimizer(object):
    """Optimize model like backpropagation."""
    
    def __init__(self, config=None, graph=None):
        """Optimize model like backpropagation.
        Args:
            config: configuration for setting optimizer.
            model: Neural network model.
        """
        
        self._config = config
        if self._config:
            self._use_autoencoder = config.use_autoencoder or False
        else:
            self._use_autoencoder = False
        self._graph = graph
    
    def _init(self, loss):
        self._ops = utility.get_train_ops(graph=self._graph)
        self.inputs, self.labels = utility.get_placeholder_ops(loss)
        self._layers = utility._zip_layer(inputs=self.inputs,
                                          loss=loss,
                                          ops=self._ops,
                                          graph=self._graph)
        # self._layers = utility.zip_layer(inputs=self.inputs,
        #                                   ops=self._ops,
        #                                   graph=self._graph)
    
    def _autoencoder(self):
        inputs_size = self._layers[0].output.shape[1]
        output = self._layers[-1].output
        self.decoder = tf.layers.Dense(inputs_size)(output)
        autoencoder_losses = tf.losses.mean_squared_error(
            labels=self.inputs, predictions=self.decoder)
        self.autoencoder_loss = tf.reduce_mean(autoencoder_losses)
        
        self.autoencoder_train_op = tf.train.AdamOptimizer().minimize(self.autoencoder_loss)
    
    def minimize(self, loss=None):
        """Construct the control dependencies for calculating neural net optimized.
        
        Returns:
            tf.no_op.
            The import
        """
        self._init(loss)
        
        if self._use_autoencoder:
            self._autoencoder()
        
        a = self.labels
        updates = []
        # Reverse
        layers = self._layers[::-1]
        for i, layer in enumerate(layers):
            u = layer.output
            v = layer.kernel
            if layer.use_bias:
                v = tf.concat((v, layer.bias[None, ...]), axis=0)
            
            # Not use activation (ReLU)
            if not layer.activation:
                u, v = mf.semi_nmf(a=a, u=u, v=v,
                                   use_tf=True,
                                   use_bias=layer.use_bias,
                                   num_iters=1,
                                   first_nneg=True,
                                   )
            # Use activation (ReLU)
            elif utility.get_op_name(layer.activation) == 'Relu':
                u, v = mf.nonlin_semi_nmf(a=a, u=u, v=v,
                                          use_tf=True,
                                          use_bias=layer.use_bias,
                                          num_calc_v=1,
                                          num_calc_u=1,
                                          first_nneg=True,
                                          )
            # Use Softmax
            elif utility.get_op_name(layer.activation) == 'Softmax':
                print('used softmax!!')
                u, v = mf.softmax_nmf(a=a, u=u, v=v,
                                      use_tf=True,
                                      use_bias=layer.use_bias,
                                      )
            if layer.use_bias:
                v, bias = utility.split_v_bias(v)
                updates.append(layer.bias.assign(bias))
            updates.append(layer.kernel.assign(v))
            a = tf.identity(u)
        
        return tf.group(*updates)