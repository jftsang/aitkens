import numpy as np


def second_differences(xs, *, direction):
    """Compute second differences. Returns three numpy arrays: the first
    and second differences, as well as a truncated version of the input
    array so that the positions at each index match up.
    """
    xs = np.array(xs)
    if direction == 'forward':
        trunc_xs = xs[:-2]
        dxs = xs[1:] - xs[:-1]
        d2xs = dxs[1:] - dxs[:-1]
        return trunc_xs, dxs[:-1], d2xs
    elif direction == 'central':
        trunc_xs = xs[1:-1]
        dxs = (xs[2:] - xs[:-2]) / 2
        d2xs = xs[2:] - 2*xs[1:-1] + xs[:-2]
        return trunc_xs, dxs, d2xs
    else:
        raise NotImplementedError('direction must be forward or central')


def accelerate(xs, *, direction='forward'):
    xs, dxs, d2xs = second_differences(xs, direction=direction)
    return np.where(
        np.logical_and(dxs == 0, d2xs == 0),
        xs,
        xs - dxs ** 2 / d2xs
    )
