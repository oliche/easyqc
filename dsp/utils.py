"""
Window generator, front detections, rms
"""
import numpy as np


def _fcn_extrap(x, f, bounds):
    """
    Extrapolates a flat value before and after bounds
    x: array to be filtered
    f: function to be applied between bounds (cf. fcn_cosine below)
    bounds: 2 elements list or np.array
    """
    y = f(x)
    y[x < bounds[0]] = f(bounds[0])
    y[x > bounds[1]] = f(bounds[1])
    return y


def fcn_cosine(bounds):
    """
    Returns a soft thresholding function with a cosine taper:
    values <= bounds[0]: values
    values < bounds[0] < bounds[1] : cosine taper
    values < bounds[1]: bounds[1]
    :param bounds:
    :return: lambda function
    """
    def _cos(x):
        return (1 - np.cos((x - bounds[0]) / (bounds[1] - bounds[0]) * np.pi)) / 2
    func = lambda x: _fcn_extrap(x, _cos, bounds)  # noqa
    return func


def fronts(x, axis=-1, step=1):
    """
    Detects Rising and Falling edges of a voltage signal, returns indices and

    :param x: array on which to compute RMS
    :param axis: (optional, -1) negative value
    :param step: (optional, -1) value of the step to detect
    :return: numpy array of indices, numpy array of rises (1) and falls (-1)
    """
    d = np.diff(x, axis=axis)
    ind = np.array(np.where(np.abs(d) >= step))
    sign = d[tuple(ind)]
    ind[axis] += 1
    if len(ind) == 1:
        return ind[0], sign
    else:
        return ind, sign
    return rises(np.abs(x), axis=axis, step=step)


def falls(x, axis=-1, step=-1):
    """
    Detects Falling edges of a voltage signal, returns indices

    :param x: array on which to compute RMS
    :param axis: (optional, -1) negative value
    :param step: (optional, -1) value of the step to detect
    :return: numpy array
    """
    return rises(-x, axis=axis, step=-step)


def rises(x, axis=-1, step=1):
    """
    Detect Rising edges of a voltage signal, returns indices

    :param x: array on which to compute RMS
    :param axis: (optional, -1)
    :param step: (optional, 1) amplitude of the step to detect
    :return: numpy array
    """
    ind = np.array(np.where(np.diff(x, axis=axis) >= step))
    ind[axis] += 1
    if len(ind) == 1:
        return ind[0]
    else:
        return ind


def rms(x, axis=-1):
    """
    Root mean square of array along axis

    :param x: array on which to compute RMS
    :param axis: (optional, -1)
    :return: numpy array
    """
    return np.sqrt(np.mean(x ** 2, axis=axis))
