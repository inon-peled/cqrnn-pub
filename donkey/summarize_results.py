import pandas as pd


def summarize_results(pkl_path, obs_set, ratio_for_reasonable_val_mil):
    return pd.read_pickle(pkl_path)\
            [lambda d: d.obs_set == obs_set]\
            [lambda d: d.mil_val / d.mean_non_cens_y_obs_val <= ratio_for_reasonable_val_mil]\
            .assign(icp_val_proximity_to_90perc=lambda d: (d.icp_val - 0.9).abs())\
            .sort_values(by=['icp_val_proximity_to_90perc', 'mil_val', 'cross_val'])\
            .groupby(['model', 'superhub', 'cens_seed', 'approx_perc_cens', 'cens_low', 'cens_high'])\
            .first()\
            .reset_index()\
            .groupby(['model', 'superhub', 'approx_perc_cens', 'cens_low', 'cens_high'])\
            .agg(['mean', 'std'])\
            [['icp_test', 'mil_test', 'cross_test']]\
            .assign(obs_set=obs_set)
