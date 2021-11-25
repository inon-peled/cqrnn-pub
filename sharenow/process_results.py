from evaluate_predictions import evaluate_predictions

import pandas as pd

import os


def process_results(obs_set, do_flip, num_lags, persist_dir, cens_seed_range, tf_seed_range, frac_range):
    def rename_dict_keys(d_eval, obs_set, what):
        return {
            ('%s_%s' % (key, what)): d_eval[obs_set][key]
            for key in d_eval[obs_set].keys()
        }
    def process_one_experiment(cens_seed, tf_seed, frac):
        q5_pred = pd.read_pickle(os.path.join(persist_dir, 'l_%d_q_%d_f_%.2f_s1_%d_s2_%d.pkl' % (num_lags, 5, frac, cens_seed, tf_seed)))
        q95_pred = pd.read_pickle(os.path.join(persist_dir, 'l_%d_q_%d_f_%.2f_s1_%d_s2_%d.pkl' % (num_lags, 95, frac, cens_seed, tf_seed)))
        return pd.concat([pd.DataFrame(rename_dict_keys(
            d_eval=evaluate_predictions(
                is_censored=q5_pred['is_censored_%s' % what],
                q5_pred=(-1.0 if do_flip else 1.0) * (q95_pred if do_flip else q5_pred)['pred_%s' % what], 
                q95_pred=(-1.0 if do_flip else 1.0) * (q5_pred if do_flip else q95_pred)['pred_%s' % what], 
                y_true=q5_pred['ystar_%s' % what].values),
            what=what,
            obs_set=obs_set), index=[0]) for what in ['train', 'val', 'test']], axis=1)
#     for cens_seed in cens_seed_range:
#         for tf_seed in tf_seed_range:
#             for frac in frac_range:
#                 print(num_lags, cens_seed, tf_seed, frac, 
#                       process_one_experiment(cens_seed=cens_seed, tf_seed=tf_seed, frac=frac))
    
    df_res = pd.concat([pd.DataFrame(
        process_one_experiment(cens_seed=cens_seed, tf_seed=tf_seed, frac=frac))\
                        .assign(frac=frac, cens_seed=cens_seed, tf_seed=tf_seed)
            for cens_seed in cens_seed_range 
            for tf_seed in tf_seed_range 
            for frac in frac_range])\
        .assign(icp_val_proximity_to_90perc=lambda d: abs(d.icp_val - 0.9))\
        .sort_values(by=['icp_val_proximity_to_90perc', 'mil_val', 'cross_val','tl_val'])\
        .groupby(['frac', 'cens_seed'])\
        .first()\
        .reset_index()\
        .groupby('frac')\
        .agg(['mean', 'std'])\
        [['icp_test', 'mil_test', 'cross_test','tl_test']]

    tab_str = ''
    for row in df_res.iterrows():
        tab_str += ('& \\ms{%.3f}{%.1f}' % (row[1]['icp_test']['mean'], row[1]['icp_test']['std'])).replace('0.', '.')
        tab_str += (' & \\ms{%d}{%d}' % (row[1]['mil_test']['mean'], row[1]['mil_test']['std']))
        tab_str += (' & \\ms{%.3f}{%.1f}' % (row[1]['cross_test']['mean'], row[1]['cross_test']['std']))
        #tab_str += (' & \\ms{%d}{%d}' % (row[1]['tl_test']['mean'], row[1]['tl_test']['std']))
    tab_str += ' \\\\'
    print(tab_str)

    return df_res
