from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def frobenius_norm(original_x, muled_y, axis=0):
    losses = original_x - muled_y
    loss = tf.norm(losses, axis=axis) / tf.norm(original_x)
    return loss


def np_frobenius_norm(original_x, muled_y):
    losses = original_x - muled_y
    loss = np.linalg.norm(losses) / np.linalg.norm(original_x)
    return loss