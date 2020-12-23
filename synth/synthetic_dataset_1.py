from generate_synthetic_data import generate_synthetic_data

import numpy as np
from scipy.stats import norm


def make_ds1(verbose, num_samples, seed):
    return generate_synthetic_data(
        verbose=verbose,
        size=num_samples,
        betas=(1, 1, 1), 
        seed=seed, 
        epsilon=epsilon_ds1)


def epsilon_ds1(X, prng, size):
    return prng.normal(size=size)
        

def ppf_ystar_ds1(ds, theta):
    return norm(loc=ds['x0'] + ds['x1'] + ds['x2'],
                scale=1)\
        .ppf(theta)


def ppf_censored_y_ds1(ds, theta):
    return np.maximum(0, ppf_ystar_ds1(ds, theta))
