import numpy as np


def accelerate(xs):
    npxs = np.array(xs)
    dxs = npxs[1:] - npxs[:-1]
    d2xs = dxs[1:] - dxs[:-1]
    return np.where(
        np.logical_and(dxs[:-1] == 0, d2xs == 0),
        npxs[:-2],
        npxs[:-2] - dxs[:-1] ** 2 / d2xs
    )
