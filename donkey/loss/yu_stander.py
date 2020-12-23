from .tilted_loss import tl

import tensorflow as tf


def make_loss_function_yu_standr_nll(theta, lower_threshold):
    return lambda y_true, y_pred:\
        tf.math.reduce_sum(tl(theta=theta, e=y_true - tf.math.maximum(lower_threshold, y_pred)))
