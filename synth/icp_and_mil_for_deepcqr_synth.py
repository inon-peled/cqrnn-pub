from synthetic_dataset_1 import make_ds1
from synthetic_dataset_2 import make_ds2
from synthetic_dataset_3 import make_ds3

from scipy.stats import norm
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

import pathlib
import sys
import pickle
from functools import partial


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
    

def _mil_icp_cross(y_true, qpred_5, qpred_95):
    return {
        'mil': np.abs(qpred_95 - qpred_5).mean(),
        'cross': (qpred_95 <= qpred_5).mean(),
        'icp': (np.logical_and(y_true >= qpred_5, y_true <= qpred_95)).mean()
    }
    
    
def _scores(noise_sigma, y_pred, y_true):
    def q_pred(q_int):
        return norm(loc=y_pred, scale=noise_sigma).ppf(q_int / 100.0)
    
    return _mil_icp_cross(y_true=y_true.values.flatten(), qpred_5=q_pred(5), qpred_95=q_pred(95))


def evaluation_measures(noise_sigma, y_pred, y_true):
    print('y_true > 0 = %d of %d (%.2f)' % (sum(y_true > 0), len(y_true), sum(y_true > 0) / len(y_true)))
    return {
        'all': _scores(noise_sigma=noise_sigma, y_pred=y_pred, y_true=y_true),
        'only_non_censored': _scores(noise_sigma=noise_sigma, y_pred=y_pred[y_true > 0], y_true=y_true[y_true > 0]),
    }


def create_single_unit_model(loss, kernel_initializer, activation, optimizer, num_x_features):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            units=1,
            use_bias=False,
            activation=activation,
            input_shape=[num_x_features],
            kernel_initializer=kernel_initializer
        )
    ])
#     print(activation, model.get_weights())
    model.compile(loss=loss, 
                  optimizer=optimizer)
    return model


def fit(seed, x_train, x_val, y_train, y_val, model,
        verbose=False, batch_size=1000, epochs=5000, validation_split=0.2, 
        early_stop_patience=10, early_stop_min_delta=0):
    tf.random.set_seed(seed)
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


def create_fit_predict_evaluate(
        noise_sigma,
        splitter, initalization_creator, fit_seeds, 
        model_creator, verbose, dataset):
    x_train, x_val, x_test, y_train, y_val, y_test, ystar_test = splitter(dataset)
    print('Fraction censored of test:', (y_test <= 0).mean())
    best_model = None
    for seed in tqdm(fit_seeds, desc='fit seeds'):
        model=model_creator(kernel_initializer=initalization_creator(seed))
        fitted_model = fit(
            seed=seed,
            x_train=x_train, 
            x_val=x_val, 
            y_train=y_train, 
            y_val=y_val,
            verbose=verbose,
            model=model
        )
        curr_model_loss = fitted_model.evaluate(x_val, y_val, batch_size=10000, verbose=False)
        best_model_loss = None if best_model is None else best_model.evaluate(x_val, y_val, batch_size=10000, verbose=False)
        print('Current loss = %s, best loss before current = %s' % (curr_model_loss, best_model_loss))
        best_model = model if best_model is None or curr_model_loss < best_model_loss else best_model        
    print('Best model weights:', best_model.get_weights())
    y_pred = fitted_model.predict(x_test, batch_size=10000).flatten()
    return {
        'ystar_pred': y_pred.tolist(),
        'ystar_true': ystar_test.tolist(), 
        'evaluation': evaluation_measures(noise_sigma=noise_sigma, y_pred=y_pred, y_true=ystar_test)
    }


def make_qr_loss_function(theta):
    def _tl(e):
        return tf.math.maximum(theta * e, (theta - 1) * e)
    
    return lambda y_true, y_pred: tf.math.reduce_sum(_tl(e=y_true - tf.math.maximum(0.0, y_pred)))


def create_initializer_near_0_or_1(seed):
    return tf.keras.initializers.RandomNormal(mean=seed % 2, stddev=0.05, seed=seed)


def get_range_of_fit_seeds():
    return range(100, 120)


def _one_theta(ds_num, theta):
    return theta, create_fit_predict_evaluate(
        splitter=partial(train_validation_test_random_splitter, seed=0, percent_test=0.33, percent_val_from_train=0.2),
        initalization_creator=create_initializer_near_0_or_1,
        fit_seeds=get_range_of_fit_seeds(),
        model_creator=partial(
            create_single_unit_model, 
            loss=make_qr_loss_function(theta=theta),
            activation='linear',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1, clipnorm=1),
            num_x_features=3),
        verbose=False,
        dataset={1: make_ds1, 2: make_ds2, 3: make_ds3}[ds_num]\
            (verbose=True, num_samples=1000, seed=42)
    )


def persist(ds_num, theta):
    obj = _one_theta(ds_num=ds_num, theta=theta)
    dirpath = os.path.join('.', 'thetas', 'synth_ds_%d' % ds_num)
    pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(dirpath, ('theta_%0.2f.pkl' % theta)), 'wb') as f:
        pickle.dump(obj=obj, file=f, protocol=3)
    

if __name__ == '__main__':
    persist(theta=float(sys.argv[1]), ds_num=int(sys.argv[2]))
