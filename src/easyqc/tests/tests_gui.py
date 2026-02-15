import numpy as np
import pytest

from qtpy import QtCore
from easyqc.gui import EasyQC, viewseis


def _ricker(points: int, a: float) -> np.ndarray:
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = 1 - xsq / wsq
    gauss = np.exp(-xsq / (2 * wsq))
    return A * mod * gauss


@pytest.fixture
def synthetic_seis():
    ntr, ns = 500, 2000
    data = np.zeros((ntr, ns), np.float32)
    data[:, 500:600] = _ricker(100, 4)

    noise = np.random.randn(ntr, ns).astype(np.float32) / 10
    a, b = np.meshgrid(np.arange(ntr / 2) * 8 + 2000, np.arange(2) * 50 + 5000)
    header = {"receiver_line": b.flatten(), "receiver_number": a.flatten()}
    return data + noise, header


@pytest.fixture
def view_with_data(qtbot, synthetic_seis):
    data, header = synthetic_seis
    w = viewseis(data, si=0.002, h=header, title="test")
    qtbot.addWidget(w)
    w.show()
    yield w
    w.close()


def test_viewseis_shows(view_with_data):
    assert view_with_data.isVisible()
    assert hasattr(view_with_data, "plotItem_seismic")


def test_window_builds(qtbot):
    w = EasyQC()
    qtbot.addWidget(w)
    w.show()

    assert w.isVisible()
    assert hasattr(w, "plotItem_seismic")


def test_gain_edit_updates(qtbot, view_with_data):
    w = view_with_data
    w.lineEdit_gain.setText("6")
    qtbot.keyPress(w.lineEdit_gain, QtCore.Qt.Key_Return)
    assert float(w.lineEdit_gain.text()) == 6.0
