from generate_synthetic_data import generate_synthetic_data

import numpy as np
from scipy.stats import norm


def make_ds3(verbose, num_samples, seed):
    return generate_synthetic_data(
        verbose=verbose,
        size=num_samples, 
        betas=(1, 1, 1), 
        seed=seed, 
        epsilon=epsilon_ds3)


def epsilon_ds3(X, prng, size):
    return (0.75 * prng.normal(size=size)) + (0.25 * prng.normal(size=size, scale=2))


def ppf_ystar_ds3(ds, theta):
    return norm(loc=ds['x0'] + ds['x1'] + ds['x2'], 
                scale=((0.75 ** 2) + (0.25 ** 2)) ** 0.5)\
        .ppf(theta)


def ppf_censored_y_ds3(ds, theta):
    return np.maximum(0, ppf_ystar_ds3(ds, theta))
