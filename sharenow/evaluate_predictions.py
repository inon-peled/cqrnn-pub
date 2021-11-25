import numpy as np
import keras.backend as K

def tilted_loss(q,e):
    return K.maximum(q*e, (q-1)*e)

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
        'tl' : K.sum(tilted_loss(0.05, y_true-q5_pred)+tilted_loss(0.95,y_true-q95_pred)).numpy()
#        'mae95': np.abs(q95_pred.flatten() - y_true.flatten()).mean()
    }
