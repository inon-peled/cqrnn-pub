import tensorflow as tf


def tl(e, theta):
    return tf.math.maximum(theta * e, (theta - 1) * e)


def make_loss_function_tilted_loss(theta, lower_threshold):
    return lambda y_true, y_pred:\
        tf.math.reduce_sum(tl(theta=theta, e=y_true - y_pred))
