from synthetic_dataset_1 import make_ds1,ppf_ystar_ds1
from synthetic_dataset_2 import make_ds2,ppf_ystar_ds2
from synthetic_dataset_3 import make_ds3,ppf_ystar_ds3

import os
import argparse
import pathlib
import sys
import pickle
from functools import partial


import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
#import tensorflow as tf
import numpy as np

## stuff for
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def train_validation_test_random_splitter(dataset, seed, percent_test, percent_val_from_train):
    x_train_and_val, x_test, y_train_and_val, y_test = train_test_split(
        pd.DataFrame(dataset['X'].transpose()), 
        pd.Series(data=dataset['y']), 
        test_size=percent_test, 
        random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_and_val, 
        y_train_and_val, 
        test_size=percent_val_from_train, 
        random_state=seed + 1)
    ystar_test = pd.Series(dataset['y_star']).iloc[y_test.index]
    return x_train, x_val, x_test, y_train, y_val, y_test, ystar_test
    

def _scores(y_pred, y_true):
    return [
        ('r2', '%.3f' % r2_score(y_pred=y_pred, y_true=y_true)),
        ('mae', '%.3f' % abs(y_pred - y_true).mean()),
        ('rmse', '%.3f' % ((y_pred - y_true) ** 2).mean() ** 0.5),        
    ]

def _mil_icp_cross(y_true, qpred_5, qpred_95):
    return {
        'mil': np.abs(qpred_95 - qpred_5).mean(),
        'cross': (qpred_95 <= qpred_5).mean(),
        'icp': (np.logical_and(y_true >= qpred_5, y_true <= qpred_95)).mean()
    }
    


def evaluation_measures(y_pred, y_true):
    print('y_true > 0 = %d of %d (%.2f)' % (sum(y_true > 0), len(y_true), sum(y_true > 0) / len(y_true)))
    return {
        'all': _scores(y_pred=y_pred, y_true=y_true),
        'only_non_censored': _scores(y_pred=y_pred[y_true > 0], y_true=y_true[y_true > 0]),
    }

def q_pred(y_pred, q_int, noise_sigma=1):
    return norm(loc=y_pred, scale=noise_sigma).ppf(q_int)

def tobit_scores(noise_sigma, y_pred, y_true):
    return _mil_icp_cross(y_true=y_true.flatten(), qpred_5=q_pred(y_pred, 5), qpred_95=q_pred(y_pred, 95))


def tobit_evaluation_measures(noise_sigma, y_pred, y_true):
    print('y_true > 0 = %d of %d (%.2f)' % (sum(y_true > 0), len(y_true), sum(y_true > 0) / len(y_true)))
    return {
        'all': tobit_scores(noise_sigma=noise_sigma, y_pred=y_pred, y_true=y_true),
        'only_non_censored': tobit_scores(noise_sigma=noise_sigma, y_pred=y_pred[y_true > 0], y_true=y_true[y_true > 0]),
    }


def create_multi_output_target(y, quantiles):
    y_ = y[:,np.newaxis]
    for _ in range(len(quantiles)-1):
        y_ = np.concatenate((y_, y[:,np.newaxis]), axis=1)
    return y_

def tilted_loss(q,e):
    return K.maximum(q*e, (q-1)*e)

def censored_multi_tilted_loss(quantiles,y,f):
    loss = 0.0
    treshold_values = tf.where(y[:,0] == 0.0, 0.0, -np.inf)
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:,k] - K.maximum(treshold_values, f[:,k]))
        loss += K.mean(K.sum(tilted_loss(q, e), axis=-1))
    return loss

def censored_model(quantiles,loss_func):
    ipt_layer = Input((3,))  
    out1 = Dense(len(quantiles), use_bias=False)(ipt_layer)

    opt= tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1)
    
    model = Model(inputs=ipt_layer, outputs=out1)
    model.compile(loss=lambda y,f: loss_func(quantiles, y, f),
                 optimizer=opt)
     
    return model

def multi_tilted_loss(quantiles,y,f):
    loss = 0.0
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:,k] - f[:,k])
        loss += K.mean(K.sum(tilted_loss(q, e), axis=-1))

    return loss

def uncensored_model(quantiles):
    ipt_layer = Input((3,))  
    out1 = Dense(len(quantiles), use_bias=False)(ipt_layer)

    opt= tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1)
    
    model = Model(inputs=ipt_layer, outputs=out1)
    model.compile(loss=lambda y,f: multi_tilted_loss(quantiles, y, f),
                 optimizer=opt)
     
    return model

def huber(error, eps=0.001):
  cond  = K.abs(error) < eps
  squared_loss = K.square(error) / (2*eps)
  linear_loss  = K.abs(error) - 0.5 * eps

  return tf.where(cond, squared_loss, linear_loss)


def huber_tilted_loss(q, e):
    e = huber(error = e)
    return K.maximum(q*e, (q-1)*e)

def huber_ramp_function(l, u):
    cond = u >= l
    return tf.where(cond, huber(u), l)

def huber_multi_tilted_loss(quantiles, y, f):
    loss = 0.0
    treshold_values = tf.where(y[:,0] == 0.0, 0.0, -np.inf)
    for k in range(len(quantiles)):
        q = quantiles[k]
        e = (y[:,k] - K.maximum(treshold_values, f[:,k]))
        loss += K.mean(K.sum(huber_tilted_loss(q, e),axis=-1))
    return loss


def tobit_type1_nll_tensorflow(y_true, y_pred, noise_sigma, lower_threshold):
    def nll_not_censored(y_true, y_pred, gamma):
        nrm = tfp.distributions.Normal(loc=0, scale=1)
        cens_labels = y_true <= 0
        return -tf.math.reduce_sum(np.log(gamma) + nrm.log_prob(gamma * tf.boolean_mask(y_true, ~cens_labels) - tf.boolean_mask(y_pred, ~cens_labels)))
    
    def nll_censored(y_pred, gamma, lower_threshold):
        nrm = tfp.distributions.Normal(loc=0, scale=1)
        cens_labels = y_true <= 0
        return -tf.math.reduce_sum(nrm.log_cdf(gamma * lower_threshold - tf.boolean_mask(y_pred, cens_labels)))
    
    return nll_not_censored(y_pred=y_pred, 
                         y_true=y_true, 
                         gamma=1 / noise_sigma) + nll_censored(y_pred=y_pred,
                     gamma=1 / noise_sigma,
                     lower_threshold=lower_threshold)

def tobit_model():
    ipt_layer = Input((3,))  
    out1 = Dense(1, use_bias=False)(ipt_layer)

    opt= tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1)
    
    model = Model(inputs=ipt_layer, outputs=out1)
    model.compile(loss=lambda y,f: tobit_type1_nll_tensorflow(y, f, noise_sigma=1, lower_threshold=0.0),
                 optimizer=opt)
     
    return model

def fit(x_train, x_val, y_train, y_val, model,
        verbose=False, batch_size=1000, epochs=5000, validation_split=0.2, 
        early_stop_patience=100, early_stop_min_delta=0):
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                verbose=True, 
                patience=early_stop_patience, 
                min_delta=early_stop_min_delta, 
                monitor='val_loss', 
                mode='min', 
                restore_best_weights=True
            )
        ],
        verbose=verbose,
        batch_size=batch_size,
        epochs=epochs
    )
    return model

def create_fit_predict_evaluate(splitter, fit_seeds,quantiles,loss, loss_func, verbose, dataset, dataseteval):
    result_dict = {}
    x_train, x_val, x_test, y_train, y_val, y_test, ystar_test = splitter(dataset)
    x_train, x_val, x_test, y_train, y_val, y_test, ystar_test = x_train.values, x_val.values, x_test.values, y_train.values, y_val.values, y_test.values, ystar_test.values
    y_train, y_val, y_test = create_multi_output_target(y_train, quantiles), create_multi_output_target(y_val, quantiles), create_multi_output_target(y_test, quantiles)	
    print(y_train.shape)
    print('Fraction censored of test:', (y_test <= 0).mean())
    tf.keras.backend.clear_session()
    
    if loss == 'tl':
        model = uncensored_model(quantiles)
    elif loss == 'tobit':
        model=tobit_model()
    else:
        model= censored_model(quantiles, loss_func)

    fitted_model = fit(
		x_train=x_train, 
		x_val=x_val, 
		y_train=y_train, 
		y_val=y_val,
		verbose=verbose,
		model=model
	)

    preds = fitted_model(x_test)
    for i, quant in enumerate(quantiles):
        print("Performance for {}-quantile".format(quantiles[i]))
        print(evaluation_measures(y_pred= preds[:,0].numpy(), y_true=ystar_test))

        result_dict[quant] = {}
        result_dict[quant]['evaluation'] = evaluation_measures(y_pred= preds[:,i].numpy(), y_true=ystar_test)
        result_dict[quant]['y_star_pred'] = preds[:,i].numpy()
        result_dict[quant]['y_star'] = ystar_test
        
        q_test = dataseteval(theta=quant, ds={'x0': x_test[:, 0], 'x1': x_test[:, 1], 'x2': x_test[:, 2]})
        result_dict[quant]['evaluation_against_quant'] = evaluation_measures(y_true = q_test, y_pred= preds[:,i].numpy())

        

    #print(_mil_icp_cross(ystar_test, preds[:,0].numpy(),preds[:,-1].numpy()))
    return result_dict


def make_qr_loss_function(theta):
    def _tl(e):
        return tf.math.maximum(theta * e, (theta - 1) * e)
    
    return lambda y_true, y_pred: tf.math.reduce_sum(_tl(e=y_true - tf.math.maximum(0.0, y_pred)))


def create_initializer_near_0_or_1(seed):
    return tf.keras.initializers.RandomNormal(mean=int(seed % 2), seed=seed)  # tf.keras.initializers.RandomNormal(mean=abs(seed % 2), seed=seed),


def get_range_of_fit_seeds():
    return range(100, 120)


def _one_theta(ds_num, theta, loss):
    return theta, create_fit_predict_evaluate(
        splitter=partial(train_validation_test_random_splitter, seed=0, percent_test=0.33, percent_val_from_train=0.2),
        fit_seeds=get_range_of_fit_seeds(),
        quantiles=theta,
        loss = loss,
        loss_func = huber_multi_tilted_loss if loss == 'huber' else censored_multi_tilted_loss,
        verbose=False,
        dataset={1: make_ds1, 2: make_ds2, 3: make_ds3}[ds_num]\
            (verbose=True, num_samples=1000, seed=42),
        dataseteval={1: ppf_ystar_ds1, 2: ppf_ystar_ds2, 3: ppf_ystar_ds3}[ds_num]
    )


def persist(ds_num, theta, loss, num_runs, name):
    tf.random.set_seed(10)
    for i in range(num_runs):
        obj = _one_theta(ds_num=ds_num, theta=theta, loss=loss)
        for th in theta:
            dirpath = os.path.join('.', 'thetas', 'synth_ds_%d' % ds_num, 'loss_' + loss)
            pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(dirpath, ('model_'+name+'_theta_{}_run_{}.pkl'.format(th,i))), 'wb') as f:
                pickle.dump(obj=obj, file=f, protocol=3)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--theta', type=eval, default=[0.5])
    parser.add_argument('--dataset', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=10)
    
    parser.add_argument('--loss', type=str, default="huber")
    args = parser.parse_args()

    # Determine if multiple quantiles are estimated.
    if len(args.theta) > 1 :
        name = 'multi'
    else:
        name = 'single'
    tf.config.run_functions_eagerly(True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    persist(theta=args.theta, ds_num=args.dataset, loss=args.loss, num_runs=args.num_runs, name = name)
