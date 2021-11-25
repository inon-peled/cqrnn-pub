from model_creator_factory import get_model_creator
from loss_function_factory import get_loss_function

import os
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
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.get_logger().setLevel(tf.logging.WARNING)
#tf.config.experimental.enable_mlir_graph_optimization()
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import argparse
from itertools import chain
import pickle
from pathlib import Path
from functools import reduce

from tqdm import tqdm
import numpy as np
import pandas as pd


def factor_for_threshold(df_train):
    return 1.0 if (len(df_train[df_train.is_censored]) == 0 or len(df_train[~df_train.is_censored]) == 0) else \
        df_train[~df_train.is_censored].y.mean() / df_train[df_train.is_censored].y.mean()


def myprint(*args, **kwargs):
    if False:
        print(*args, **kwargs)


def get_df(superhub):
    return pd.read_csv(('../data/donkey/%s_per_superhub_%s/%s_superhub_%s.csv' % 
                        ('demand', '1D', 'demand', superhub)), 
                       parse_dates=['t']).set_index('t').squeeze()


def get_censored(seed, superhub, approx_percent_censored, censorship_intensity_low, censorship_intensity_high):
    def rand_intens(df):
        return censorship_intensity_low + (
            np.random.RandomState(seed=seed + 1).rand(len(df)) * 
            (censorship_intensity_high - censorship_intensity_low)
        )
    
    return get_df(superhub=superhub)\
        .rename('ystar')\
        .reset_index()\
        .assign(is_censored=lambda d: np.random\
                .RandomState(seed=seed)\
                .binomial(size=len(d), n=1, p=approx_percent_censored)\
                .astype(np.bool))\
        .assign(y=lambda d: (\
                (~d.is_censored) * d.ystar) + \
                (d.is_censored * ((1 - rand_intens(d)) * d.ystar)).astype(np.float64))\
        .set_index('t')


def add_lags_and_bias(df, num_lags):
    for lag in range(1, num_lags + 1):
        df = pd.merge(
            how='inner',
            right=df.drop(columns=['ystar']).shift(lag).rename(columns=dict(y='x_y%d' % lag))[['x_y%d' % lag]],
            left=df,
            left_index=True,
            right_index=True)
    return df\
        .dropna()\
        .assign(x_bias=1)


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
        'df_train': df_train,
        'df_test': df_test,
        'df_val': df_val,
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


def fit(tf_seed, x_train, x_val, y_train, y_val, model,
        verbose, batch_size, epochs, early_stop_patience, early_stop_min_delta, train_lower_threshold, val_lower_threshold):
    tf.random.set_seed(tf_seed)
    quantiles = [0.05, 0.95]
    y_train, y_val = create_multi_output_target(y_train, quantiles), create_multi_output_target(y_val, quantiles)
    train_lower_threshold = np.array(train_lower_threshold)
    val_lower_threshold = np.array(val_lower_threshold)
    y_train = np.concatenate((y_train, train_lower_threshold[:,np.newaxis]), axis=1)
    y_val = np.concatenate((y_val, val_lower_threshold[:,np.newaxis]), axis=1)
    training_history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        callbacks=[
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
#     print(model.summary())
    return model, training_history


def measure(q5_pred, q95_pred, y_true):
    return {
        'mil': np.abs(q95_pred.flatten() - q5_pred.flatten()).mean(),
        'cross': (q95_pred.flatten() <= q5_pred.flatten()).mean(),
        'icp': (np.logical_and(y_true.flatten() >= q5_pred.flatten(), y_true.flatten() <= q95_pred.flatten())).mean()
    }


def evaluate_predictions(is_censored, q5_pred, q95_pred, y_true):
    return {
        'all': measure(q5_pred=q5_pred, q95_pred=q95_pred, y_true=y_true),
        'only_non_censored': measure(q5_pred=q5_pred[~is_censored], q95_pred=q95_pred[~is_censored], y_true=y_true[~is_censored]),
        'only_censored': measure(q5_pred=q5_pred[is_censored], q95_pred=q95_pred[is_censored], y_true=y_true[is_censored])
    }


def experiment(
        verbose,
        epochs,
        num_lags,
        add_dim,
        loss_function_maker,
        model_creator,
        do_flip,
        cens_seed, 
        approx_percent_censored, 
        censorship_intensity_low, 
        censorship_intensity_high, 
        persist_dir, 
        superhub, 
        tf_seed, 
        do_plots):
    def _add_dim(arr2d):
        return np.expand_dims(arr2d, -1) if add_dim else arr2d           
    
    def _get_data():
        return add_lags_and_bias(
            df=get_censored(
                seed=cens_seed,
                superhub=superhub, 
                approx_percent_censored=approx_percent_censored, 
                censorship_intensity_low=censorship_intensity_low, 
                censorship_intensity_high=censorship_intensity_high), 
            num_lags=num_lags
        )

    def _merge_dicts(d_iter):
        return reduce(lambda d1, d2: dict(chain(d1.items(), d2.items())), d_iter)
    
    def _persist(q_int, pred, split_data):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(
            persist_dir, 
            '%s_l_%d_q_%d_a_%.2f_il_%.2f_ih_%.2f_sc_%d_st_%d.pkl' % \
                (superhub, num_lags, q_int, approx_percent_censored, censorship_intensity_low, censorship_intensity_high, cens_seed, tf_seed)), 'wb') as f_pkl:
            pickle.dump(obj=_merge_dicts([pred, split_data]), file=f_pkl)
    
    def one_theta(quantiles, split_data):
        model = model_creator(
            loss_function_maker=loss_function_maker,
#             dense_kernel_initializer=tf.keras.initializers.constant(1),
            dense_kernel_initializer=tf.keras.initializers.glorot_normal(seed=tf_seed),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1, clipnorm=1.0), 
            theta=quantiles, 
            lower_threshold=(
                (-1.0 if do_flip else 1.0) * \
                _get_data().y * \
                factor_for_threshold(df_train=split_data['df_train']))\
                .astype(np.float32)
        )
        lower_thresholds=((-1.0 if do_flip else 1.0) * _get_data().y).astype(np.float32)
        fitted_model, training_history = fit(
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
        return fitted_model   
        #return {'pred_train': fitted_model.predict(_add_dim((-1.0 if do_flip else 1.0) * split_data['x_train'].values), batch_size=1000).flatten(),
        #        'pred_val': fitted_model.predict(_add_dim((-1.0 if do_flip else 1.0) * split_data['x_val'].values), batch_size=1000).flatten(),
        #        'pred_test': fitted_model.predict(_add_dim((-1.0 if do_flip else 1.0) * split_data['x_test'].values), batch_size=1000).flatten()
        #       }

    split_data = split_and_shuffle(shuffle_seed=cens_seed + 100, df=_get_data())
    quantiles = [0.05, 0.95]
    best_model = one_theta(quantiles=quantiles, split_data=split_data)

    preds = {'pred_train': best_model.predict(_add_dim((-1.0 if do_flip else 1.0) * split_data['x_train'].values), batch_size=1000),
            'pred_val': best_model.predict(_add_dim((-1.0 if do_flip else 1.0) * split_data['x_val'].values), batch_size=1000),
            'pred_test': best_model.predict(_add_dim((-1.0 if do_flip else 1.0) * split_data['x_test'].values), batch_size=1000),
            }
    q5_pred = {'pred_train': preds['pred_train'][:,0].flatten(),
            'pred_val': preds['pred_val'][:,0].flatten(),
            'pred_test': preds['pred_test'][:,0].flatten()
            }
    q95_pred = {'pred_train': preds['pred_train'][:,1].flatten(),
            'pred_val': preds['pred_val'][:,1].flatten(),
            'pred_test': preds['pred_test'][:,1].flatten()
            }

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


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cs', type=int, help='cens_seed')
    parser.add_argument('-a', type=float, help='approx_percent_censored')
    parser.add_argument('-cl', type=float, help='censorship_intensity_low')
    parser.add_argument('-ch', type=float, help='censorship_intensity_high')
    parser.add_argument('-p', type=str, help='persist_dir')
    parser.add_argument('-s', type=str, help='superhub')
    parser.add_argument('-ts', type=int, help='tf_seed')
    parser.add_argument('-l', type=str, help='loss_function_maker')
    parser.add_argument('-m', type=str, help='model_creator')
    parser.add_argument('-f', type=int, help='do_flip')
    parser.add_argument('-n', type=int, help='num_lags')
    parser.add_argument('-e', type=int, help='epochs')
    parser.add_argument('-d', type=int, help='add_dim')
    parser.add_argument('-v', type=int, help='verbose')
    args = parser.parse_args()
    experiment(
        cens_seed=args.cs, 
        approx_percent_censored=args.a, 
        censorship_intensity_low=args.cl, 
        censorship_intensity_high=args.ch, 
        persist_dir=args.p, 
        superhub=args.s, 
        tf_seed=args.ts, 
        do_plots=False,
        add_dim=args.d,
        loss_function_maker=get_loss_function(args.l),
        model_creator=get_model_creator(args.m),
        do_flip=args.f,
        num_lags=args.n,
        epochs=args.e,
        verbose=args.v
    )
    

if __name__ == '__main__':
    parse_cli()
