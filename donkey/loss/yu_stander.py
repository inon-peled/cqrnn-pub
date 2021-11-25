from .tilted_loss import tl

import tensorflow as tf
import numpy as np
import keras.backend as K

def make_loss_function_yu_standr_nll(theta, lower_threshold):
    return lambda y_true, y_pred:\
        tf.math.reduce_sum(tl(theta=theta, e=y_true - tf.math.maximum(lower_threshold, y_pred)))


def censored_multi_tilted_loss(quantiles, y, f):
    loss = 0.0
    treshold_values = tf.where(y[:,0] > y[:,-1], y[:,-1], -np.inf)
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:, k] - K.maximum(treshold_values, f[:,k]))
        loss += K.mean(K.sum(tl(theta=q, e=e), axis=-1))
    return loss

def make_loss_function_yu_standr_nll_multi(theta, lower_threshold):
    quantiles = [0.05, 0.95]
    return lambda y_true, y_pred:  tf.math.reduce_sum(censored_multi_tilted_loss(quantiles, y_true, y_pred))