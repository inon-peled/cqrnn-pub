from loss_function_factory import loss_function_factory
from model_creator_factory import model_creator_factory

import os
# Use CPU, not GPU:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Try to limit to one thread:
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLOSC_NTHREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from haversine import haversine, haversine_vector
import seaborn as sns
import pandas as pd
from pandas.plotting import register_matplotlib_converters

import sys
from itertools import chain
import pickle
from pathlib import Path
from functools import partial, reduce


def myprint(*args, **kwargs):
    if False:
        print(*args, **kwargs)

        
def daily_pickups(df):
    return df['Begin Time']\
        .dt\
        .date\
        .value_counts()\
        .astype(np.float32)\
        .sort_index()\
        .reset_index()\
        .rename(columns={'index': 'date'})\
        [lambda d: pd.to_datetime(d['date']).dt.year > 2015]


def censor_by_taking_off_random_vehicles(df, seed, fraction_of_all_vehicles):
    rand_vehicles = pd.Series(df.VIN.unique()).sample(random_state=seed, frac=fraction_of_all_vehicles)
    return pd.merge(
        how='left',
        left=daily_pickups(df[~df.VIN.isin(rand_vehicles)]),
        right=daily_pickups(df),
        on='date'
    ).rename(columns={'Begin Time_x': 'y', 'Begin Time_y': 'ystar'})


def add_time_series_features(df, num_lags):
    for lag in range(1, num_lags + 1):
        df = pd.merge(
            how='inner',
            right=df.drop(columns=['ystar']).shift(lag).rename(columns=dict(y='x_y%d' % lag))[['x_y%d' % lag]],
            left=df,
            left_index=True,
            right_index=True)
    df = df        .dropna()#         .assign(x_bias=np.float32(1))
      
    return df


def split_and_shuffle(df, shuffle_seed):
#     def is_similar_censorship_percent(df1, df2):
#         myprint(df1.is_censored.mean(), df2.is_censored.mean())
#         return abs(df1.is_censored.mean() - df2.is_censored.mean()) <= 0.03
    
    df_train = df[:len(df) // 3]\
        .sample(frac=1, random_state=shuffle_seed)
    df_val = df[len(df) // 3 : 2 * len(df) // 3]\
        .sample(frac=1, random_state=shuffle_seed + 1)
    df_test = df[2 * len(df) // 3:]
    x_cols = [c for c in df.columns if c.startswith('x_')]
    myprint('x columns are:', x_cols)
    myprint('df_train head:', df_train.head())
    myprint('df_val head:', df_val.head())
    myprint('df_test head:', df_test.head())
#     assert is_similar_censorship_percent(df_train, df_test) and \
#         is_similar_censorship_percent(df_train, df_val) and \
#         is_similar_censorship_percent(df_test, df_val)
    splt = {
        'is_censored_test': df_test.ystar != df_test.y,
        'is_censored_train': df_train.ystar != df_train.y,
        'is_censored_val': df_val.ystar != df_val.y,
        'ystar_train': df_train.ystar,
        'ystar_val': df_val.ystar,
        'ystar_test': df_test.ystar,
        'idx_train': df_train.index,
        'idx_test': df_test.index,
        'idx_val': df_val.index,
        'x_train': df_train[x_cols], 
        'x_val': df_val[x_cols], 
        'x_test': df_test[x_cols], 
        'y_train': df_train.y, 
        'y_val': df_val.y, 
        'y_test': df_test.y
    }
    for key in splt:
        myprint(key, splt[key][:2])
    return splt

def create_multi_output_target(y, quantiles):
    y__ = np.array(y)
    y_ = y__[:,np.newaxis]
    for _ in range(len(quantiles)-1):
        y_ = np.concatenate((y_, y__[:,np.newaxis]), axis=1)
    return y_

def fit(additional_callbacks, tf_seed, x_train, x_val, y_train, y_val, model,
        verbose, batch_size, epochs, early_stop_patience, early_stop_min_delta, train_lower_threshold, val_lower_threshold):
    tf.random.set_seed(tf_seed)
    training_history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        callbacks=additional_callbacks + [
            tf.keras.callbacks.EarlyStopping(
                verbose=verbose, 
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
#     myprint(model.summary())
#    for i, layer in enumerate(model.layers):
#        myprint(i, layer.get_weights()[0])
    return model, training_history


def plot_training_history(title, history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(history.history['loss'])), history.history['loss'], 
            label='Train Loss', color='blue', alpha=0.66, linestyle='-', linewidth=3)
    ax.plot(range(len(history.history['val_loss'])), history.history['val_loss'], 
            label='Validation Loss', color='red', alpha=0.66, linestyle='--', linewidth=3)
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    return fig


def plot_qr(title, idx, q5_pred, q95_pred, ystar, y):
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.scatter(idx, ystar, label='$y^*$ (latent)', color='blue', alpha=0.8, marker='o', zorder=1)
    ax.scatter(idx, y, label='$y$ (observed)', color='red', alpha=0.8, marker='x', zorder=1)
    ax.plot(idx, q5_pred, label='q5', color='dimgrey', alpha=0.8, linestyle='--', linewidth=3, zorder=2)
    ax.plot(idx, q95_pred, label='q95', color='black', alpha=0.8, linestyle='-', linewidth=3, zorder=2)
    ax.legend()
    ax.set_title(title)
    return fig


def measure(q5_pred, q95_pred, y_true):
    return {
        'mil': np.abs(q95_pred.flatten() - q5_pred.flatten()).mean(),
        'cross': (q95_pred.flatten() <= q5_pred.flatten()).mean(),
        'icp': (np.logical_and(y_true.flatten() >= q5_pred.flatten(), y_true.flatten() <= q95_pred.flatten())).mean(),
    }

def evaluate_predictions(is_censored, q5_pred, q95_pred, y_true):
    return {
        # because of our censorship scheme, "all" is the same as "only_censored".
        'all': measure(q5_pred=q5_pred, q95_pred=q95_pred, y_true=y_true),
#         'only_non_censored': measure(q5_pred=q5_pred[~is_censored], q95_pred=q95_pred[~is_censored], y_true=y_true[~is_censored]),
#         'only_censored': measure(q5_pred=q5_pred[is_censored], q95_pred=q95_pred[is_censored], y_true=y_true[is_censored])
    }


def experiment(add_dim, loss_function_maker, model_creator, do_flip, 
               callbacks, epochs, verbose, num_lags, cens_seed, fraction_of_all_vehicles, 
               persist_dir, tf_seed, do_plots):
    def _add_dim(arr2d):
        return np.expand_dims(arr2d, -1) if add_dim else arr2d           
    
    def _get_data():
        return add_time_series_features(
            df=censor_by_taking_off_random_vehicles(
                df=pd.read_parquet('../data/drivenow/df_all.parq'), 
                seed=cens_seed, 
                fraction_of_all_vehicles=fraction_of_all_vehicles), 
            num_lags=num_lags)
        
    def _merge_dicts(d_iter):
        return reduce(lambda d1, d2: dict(chain(d1.items(), d2.items())), d_iter)
    
    def _persist(q_int, pred, split_data):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(
            persist_dir, 
            'l_%d_q_%d_f_%.2f_s1_%d_s2_%d.pkl' % \
                (num_lags, q_int, fraction_of_all_vehicles, cens_seed, tf_seed)), 'wb') as f_pkl:
            pickle.dump(obj=_merge_dicts([pred, split_data]), file=f_pkl)
    
    def one_theta(theta, split_data):
        lower_thresholds=((-1.0 if do_flip else 1.0) * _get_data().y).astype(np.float32)
        model = model_creator(
            loss_function_maker=loss_function_maker,
#             dense_kernel_initializer=tf.keras.initializers.constant(1),
            dense_kernel_initializer=tf.keras.initializers.glorot_normal(seed=tf_seed),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1, clipnorm=1.0), 
            theta=theta, 
            lower_threshold=lower_thresholds)
        fitted_model, training_history = fit(
            additional_callbacks=callbacks,
            early_stop_patience=10,
            early_stop_min_delta=0,
            batch_size=100,
            epochs=epochs,
            verbose=verbose, 
            tf_seed=tf_seed,
            x_train=_add_dim((-1.0 if do_flip else 1.0) * split_data['x_train']), 
            x_val=_add_dim((-1.0 if do_flip else 1.0) * split_data['x_val']), 
            y_train=(-1.0 if do_flip else 1.0) * split_data['y_train'], 
            y_val=(-1.0 if do_flip else 1.0) * split_data['y_val'], 
            model=model,
            train_lower_threshold = lower_thresholds[split_data['y_train'].index],
            val_lower_threshold = lower_thresholds[split_data['y_val'].index])
        if do_plots:
            plot_training_history(
                title=('$\\theta = %.2f$' % theta), 
                history=training_history
            )
        return {'pred_train': fitted_model.predict(_add_dim((-1.0 if do_flip else 1.0) * split_data['x_train'].values), batch_size=1000).flatten(),
                'pred_val': fitted_model.predict(_add_dim((-1.0 if do_flip else 1.0) * split_data['x_val'].values), batch_size=1000).flatten(),
                'pred_test': fitted_model.predict(_add_dim((-1.0 if do_flip else 1.0) * split_data['x_test'].values), batch_size=1000).flatten()
               }

    split_data = split_and_shuffle(shuffle_seed=cens_seed + 100, df=_get_data())
    q5_pred = one_theta(theta=0.05, split_data=split_data)
    q95_pred = one_theta(theta=0.95, split_data=split_data)
    _persist(q_int=5, pred=q5_pred, split_data=split_data)
    _persist(q_int=95, pred=q95_pred, split_data=split_data)
    
    # Note: we next purposely mirror q5 and q95, to revert the negation (-1.0 *).
    if do_plots:
        for what in ['train', 'val', 'test']:
            plot_qr(title=what.title(),
                    idx=split_data['idx_%s' % what],
                    q5_pred=(-1.0 if do_flip else 1.0) * (q95_pred if do_flip else q5_pred)['pred_%s' % what],
                    q95_pred=(-1.0 if do_flip else 1.0) * (q5_pred if do_flip else q95_pred)['pred_%s' % what], 
                    y=split_data['y_%s' % what].values,
                    ystar=split_data['ystar_%s' % what].values)
    return {
        what: evaluate_predictions(
            is_censored=split_data['is_censored_%s' % what],
            q95_pred=(-1.0 if do_flip else 1.0) * (q5_pred if do_flip else q95_pred)['pred_%s' % what], 
            q5_pred=(-1.0 if do_flip else 1.0) * (q95_pred if do_flip else q5_pred)['pred_%s' % what], 
            y_true=split_data['ystar_%s' % what].values
        )
        for what in ['train', 'val', 'test']
    }


def parse_cli(av):
    experiment(
        num_lags=int(av[1]),
        cens_seed=int(av[2]),
        fraction_of_all_vehicles=float(av[3]),
        tf_seed=int(av[4]),
        epochs=int(av[5]),
        do_flip=bool(int(av[6])),
        persist_dir=av[7],
        model_creator=model_creator_factory[av[8]],
        loss_function_maker=loss_function_factory[av[9]],
        add_dim=bool(int(av[10])),
        callbacks=[],
        verbose=False,
        do_plots=False)

if __name__ == '__main__':
    parse_cli(sys.argv)
