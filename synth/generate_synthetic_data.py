import numpy as np
from numpy.random import RandomState


def generate_synthetic_data(verbose, betas, seed, epsilon, size):
    prng = RandomState(seed=seed)
    x0 = np.ones(size)
    x1 = np.power(-1, prng.binomial(n=1, p=0.5, size=size))
    x2 = prng.normal(size=size)
    X = np.concatenate([x0, x1, x2]).reshape(3, size)
    noise = epsilon(X=X, prng=prng, size=size)
    y_star = np.dot(np.array(betas), X) + noise
    y = np.clip(y_star, a_min=0, a_max=None)
    if verbose:
        print('Censored observations: %d of %d (%.2f%%)' % (sum(y == 0), size, sum(y == 0) / size))
    return {'x0': x0, 'x1': x1, 'x2': x2, 'noise': noise, 'X': X, 'y': y, 'y_star': y_star}
