import numpy as np

import processing as proc


def test_agc():
    x = np.random.rand(20, 2500)
    si = .002
    x[:, 500] = 10
    x = proc.bp(x, si, b=[5, 10, 40, 80])
    x_, gain = proc.agc(x, wl=.5, si=si)
    assert(np.all(x_[:, 500] / x[:, 500] > 2))
    # import easyqc
    # easyqc.viewdata(x, si)
    # easyqc.viewdata(x_, si, 'agc')


def test_fk():
    x = np.random.rand(20, 2500)
    proc.fk(x, si=.002, dx=10, vbounds=[2500, 4000], pad=0.2, lagc=0.005)
