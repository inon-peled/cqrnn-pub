import numpy as np
import keras.backend as K

def tilted_loss(q,e):
    return np.maximum(q*e, (q-1)*e)

def evaluate_predictions(is_censored, q5_pred, q95_pred, y_true):
    return {
        'all': _measure(q5_pred=q5_pred, q95_pred=q95_pred, y_true=y_true),
        'only_non_censored': _measure(q5_pred=q5_pred[~is_censored], q95_pred=q95_pred[~is_censored], y_true=y_true[~is_censored]),
        'only_censored': _measure(q5_pred=q5_pred[is_censored], q95_pred=q95_pred[is_censored], y_true=y_true[is_censored])
    }

def _measure(q5_pred, q95_pred, y_true):
    return {
        'mil': np.nan,
        'cross': np.nan,
        'icp': np.nan,
        'tl': np.nan
    } if (q95_pred.flatten() < q5_pred.flatten()).all() else {
        'mil': np.abs(q95_pred.flatten() - q5_pred.flatten()).mean(),
        'cross': np.sum(np.max(0,q95_pred.flatten() - q5_pred.flatten())),
        'icp': (np.logical_and(y_true.flatten() >= q5_pred.flatten(), y_true.flatten() <= q95_pred.flatten())).mean(),
        'tl' : np.sum(tilted_loss(0.05, y_true.flatten()-q5_pred.flatten())+tilted_loss(0.95,y_true.flatten()-q95_pred.flatten()))
    }
