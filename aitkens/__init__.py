import numpy as np


def accelerate(xs, *, direction='central'):
    xs = np.array(xs)
    if direction == 'forward':
        dxs = xs[1:] - xs[:-1]
        d2xs = dxs[1:] - dxs[:-1]
        return np.where(
            np.logical_and(dxs[:-1] == 0, d2xs == 0),
            xs[:-2],
            xs[:-2] - dxs[:-1] ** 2 / d2xs
        )

    elif direction == 'central':
        dxs = (xs[2:] - xs[:-2]) / 2
        d2xs = (xs[2:] - 2*xs[1:-1] + xs[:-2]) / 2
        return np.where(
            np.logical_and(dxs == 0, d2xs == 0),
            xs[1:-1],
            xs[1:-1] - dxs ** 2 / d2xs
        )

    else:
        raise NotImplementedError('direction must be forward or central')
