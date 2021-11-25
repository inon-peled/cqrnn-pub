import tensorflow as tf
import keras.backend as K


def tl(e, theta):
    return tf.math.maximum(theta * e, (theta - 1) * e)


def make_loss_function_tilted_loss(theta, lower_threshold):
    return lambda y_true, y_pred:\
        tf.math.reduce_sum(tl(theta=theta, e=y_true - y_pred))


def multi_tilted_loss(quantiles,y,f):
    loss = 0.0
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:,k] - f[:,k])
        loss += K.mean(K.sum(tl(q, e), axis=-1))

    return loss

def make_loss_function_tilted_loss_multi(theta, lower_threshold):
    quantiles = [0.05, 0.95]
    return lambda y_true, y_pred:  tf.math.reduce_sum(multi_tilted_loss(quantiles, y_true, y_pred))