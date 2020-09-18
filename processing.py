import numpy as np

from dsp.utils import fcn_cosine
import dsp.fourier as ft
from dsp.fourier import bp, hp, lp  # noqa


def agc(x, wl=.5, si=.002, epsilon=1e-8):
    """
    Automatic gain control
    :param x: seismic array (sample last dimension)
    :param wl: window length (secs)
    :param si: sampling interval (secs)
    :param epsilon: whitening (useful mainly for synthetic data)
    :return:
    """
    ns_win = np.round(wl / si / 2) * 2 + 1
    w = np.hanning(ns_win)
    w /= np.sum(w)
    gain = np.sqrt(ft.convolve(x ** 2, w, mode='same'))
    gain += (np.sum(gain, axis=1) * epsilon / x.shape[-1])[:, np.newaxis]
    gain = 1 / gain
    return x * gain, gain


def fk(x, si=.002, dx=1, vbounds=None, pad=.2, lagc=.5):
    """
    Frequency-wavenumber filter: filters apparent plane-waves velocity
    :param x: the input array to be filtered
    :param si: sampling interval (secs)
    :param dx: spatial interval (usually meters)
    :param vbounds: velocity high pass [v1, v2], cosine taper from 0 to 1 between v1 and v2
    :param pad: padding ratio: will add ntraces * pad empty traces each side
    :param lagc: length of agc in seconds. If set to None or 0, no agc
    :return:
    """
    assert vbounds
    nx, nt = x.shape

    # pad with zeros left and right
    npad = int(np.round(nx * pad))
    nxp = nx + npad * 2

    # compute frequency wavenumber scales and deduce the velocity filter
    fscale = ft.fscale(nt, si)
    kscale = ft.fscale(nxp, dx)
    kscale[0] = 1e-6
    v = fscale[np.newaxis, :] / kscale[:, np.newaxis]
    fk_att = (1 - fcn_cosine(vbounds)(np.abs(v)))

    # import matplotlib.pyplot as plt
    # plt.imshow(np.fft.fftshift(np.abs(v), axes=0).T, aspect='auto', vmin=0, vmax=1e5,
    #            extent=[np.min(kscale), np.max(kscale), 0, np.max(fscale) * 2])
    # plt.imshow(np.fft.fftshift(np.abs(fk_att), axes=0).T, aspect='auto', vmin=0, vmax=1,
    #            extent=[np.min(kscale), np.max(kscale), 0, np.max(fscale) * 2])

    # apply the attenuation in fk-domain
    if not lagc:
        xf = np.copy(x)
        gain = 1
    else:
        xf, gain = agc(x, wl=lagc, si=si)
    xf = np.r_[np.zeros((npad, nt), dtype=x.dtype), xf, np.zeros((npad, nt), dtype=x.dtype)]
    xf = np.real(np.fft.ifft2(fk_att * np.fft.fft2(xf)))
    if npad > 0:
        xf = xf[npad:-npad, :]
    return xf / gain
