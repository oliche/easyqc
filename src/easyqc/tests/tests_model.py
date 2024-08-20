import numpy as np
from easyqc.gui import Model
import scipy.signal


def _synthetic_data(ntr=500, ns=2000):
    # dx, v1 = (5, 2000)
    data = np.zeros((ntr, ns), np.float32)
    data[:, 500:600] = scipy.signal.ricker(100, 4)

    # create a record with 400 traces and 2500 samples
    noise = np.random.randn(ntr, ns) / 10
    # create an arbitrary layout of 2 receiver lines of 200 sensors
    a, b = np.meshgrid(np.arange(ntr / 2) * 8 + 2000, np.arange(2) * 50 + 5000)
    # the header is a dictionary of numpy arrays, each entry being the same length as the number of traces
    header = {'receiver_line': b.flatten(), 'receiver_number': a.flatten()}

    return data + noise, header


def test_get_trace():
    ntr = 400
    ns = 2500
    si = .002

    data, header = _synthetic_data(ntr=ntr, ns=ns)
    model = Model(data=data, si=si, header=header)
    model.set_data(data=data, si=si, header=header)

    assert (np.all(model.get_trace(50.02, neighbors=0).T == model.data[50, :]))
    assert (np.all(model.get_trace(0.02, neighbors=1).T == model.data[:2, :]))
    assert (np.all(model.get_trace(40.02, neighbors=1).T == model.data[39:42, :]))
