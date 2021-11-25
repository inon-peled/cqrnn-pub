from re import U
import tensorflow as tf
import keras.backend as K
import numpy as np

def huber(epsilon, u):
    option1 = 1/2*u**2
    option2 = epsilon*(u - (0.5*epsilon))
    return tf.where(K.abs(u) <= epsilon, option1, option2)

def huber_tilted_loss(theta, e):
    return K.maximum(theta*huber(2, e), (theta-1)*huber(2,e))

def make_huber_tilted_loss(theta, lower_threshold):
    return lambda y_true, y_pred: \
            tf.math.reduce_sum(huber_tilted_loss(theta=theta, e=y_true - tf.math.maximum(lower_threshold, y_pred)))

def huber_multi_tilted_loss(quantiles, y, f):
    loss = 0.0
    treshold_values = tf.where(y[:,0] > y[:,-1], y[:,-1], -np.inf)
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:,k]-tf.math.maximum(treshold_values, f[:,k]))
        loss += K.mean(K.sum(huber_tilted_loss(theta=q, e=e),axis=-1))
    return loss

def make_huber_multi_tilted_loss(theta, lower_threshold):
    quantiles = np.array([0.05, 0.95])
    return lambda y_true, y_pred: huber_multi_tilted_loss(quantiles, y_true, y_pred)

    