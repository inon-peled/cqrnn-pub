import pandas as pd
import os
from matplotlib import pyplot as plt


def plot_qr(do_flip, num_lags, superhub, persist_dir, cens_low, cens_high, approx_perc_cens, cens_seed, tf_seed):
    q5_pkl = _get_pkl(num_lags, superhub, persist_dir, cens_low, cens_high, approx_perc_cens, 5, cens_seed, tf_seed)
    idx = q5_pkl['df_test'].index
    ystar = q5_pkl['df_test'].ystar.values
    y = q5_pkl['df_test'].y.values
    q95_pkl = _get_pkl(num_lags, superhub, persist_dir, cens_low, cens_high, approx_perc_cens, 95, cens_seed, tf_seed)
    q5_pred = (-1.0 if do_flip else 1.0) * (q95_pkl if do_flip else q5_pkl)['pred_test']
    q95_pred = (-1.0 if do_flip else 1.0) * (q5_pkl if do_flip else q95_pkl)['pred_test']
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.scatter(idx, ystar, label='$y^*$ (latent)', color='blue', alpha=1.0, marker='o', zorder=1, s=40)
    ax.scatter(idx, y, label='$y$ (observed)', color='red', alpha=1.0, marker='x', zorder=1, s=40)
    ax.plot(idx, q5_pred, label='q5', color='black', alpha=0.8, linestyle='--', linewidth=3, zorder=2)
    ax.plot(idx, q95_pred, label='q95', color='black', alpha=0.8, linestyle='-', linewidth=3, zorder=2)
    ax.legend()
    ax.set_title('%s, %d lags, $(c_1, c_2) = (%.2f, %.2f$), $\gamma=%.2f$' % \
                 (superhub, num_lags, cens_low, cens_high, approx_perc_cens), fontsize=16)


def _get_pkl(num_lags, superhub, persist_dir, cens_low, cens_high, approx_perc_cens, q_int, cens_seed, tf_seed):
    return pd.read_pickle(
        os.path.join(
            persist_dir, 
            '%s_l_%d_q_%d_a_%.2f_il_%.2f_ih_%.2f_sc_%d_st_%d.pkl' % \
            (superhub, num_lags, q_int, approx_perc_cens, 
             cens_low, cens_high, cens_seed, tf_seed)
        )
    )
