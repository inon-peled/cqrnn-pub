import numpy as np

def evaluate_predictions(is_censored, q5_pred, q95_pred, y_true):
    return {
        # because of our censorship scheme, "all" is the same as "only_censored".
        'all': _measure(q5_pred=q5_pred, q95_pred=q95_pred, y_true=y_true),
#         'only_non_censored': _measure(q5_pred=q5_pred[~is_censored], q95_pred=q95_pred[~is_censored], y_true=y_true[~is_censored]),
#         'only_censored': _measure(q5_pred=q5_pred[is_censored], q95_pred=q95_pred[is_censored], y_true=y_true[is_censored])
    }


def _measure(q5_pred, q95_pred, y_true):
    return {
        'mil': np.abs(q95_pred.flatten() - q5_pred.flatten()).mean(),
        'cross': (q95_pred.flatten() <= q5_pred.flatten()).mean(),
        'icp': (np.logical_and(y_true.flatten() >= q5_pred.flatten(), y_true.flatten() <= q95_pred.flatten())).mean(),
#        'mae95': np.abs(q95_pred.flatten() - y_true.flatten()).mean()
    }
