import numpy as np
import pytest

from qtpy import QtCore
from easyqc.gui import EasyQC, viewseis


def ricker(points: int, a: float) -> np.ndarray:
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
    data[:, 500:600] = ricker(100, 4)

    noise = np.random.randn(ntr, ns).astype(np.float32) / 10
    a, b = np.meshgrid(np.arange(ntr / 2) * 8 + 2000, np.arange(2) * 50 + 5000)
    header = {"receiver_line": b.flatten(), "receiver_number": a.flatten()}
    return data + noise, header


@pytest.fixture
def easyqc_window(qtbot):
    w = EasyQC()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    yield w
    w.close()
    w.deleteLater()
    qtbot.wait(50)


@pytest.fixture
def view_with_data(qtbot, synthetic_seis):
    data, header = synthetic_seis
    w = viewseis(data, si=0.002, h=header, title="test")
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    yield w
    w.close()
    w.deleteLater()
    qtbot.wait(50)


def test_viewseis_shows(view_with_data):
    assert view_with_data.isVisible()
    assert hasattr(view_with_data, "plotItem_seismic")


def test_window_builds(easyqc_window):
    assert easyqc_window.isVisible()
    assert hasattr(easyqc_window, "plotItem_seismic")


def test_gain_edit_updates(view_with_data, qtbot):
    w = view_with_data
    w.lineEdit_gain.setText("6")
    qtbot.keyPress(w.lineEdit_gain, QtCore.Qt.Key_Return)
    qtbot.mouseClick(w.radio_wiggle, QtCore.Qt.LeftButton)
    qtbot.keyPress(w.lineEdit_gain, QtCore.Qt.Key_Return)
    assert float(w.lineEdit_gain.text()) == 6.0


def test_toggle_density_wiggle(view_with_data, qtbot):
    w = view_with_data
    assert w._display_mode == "density"
    qtbot.mouseClick(w.radio_wiggle, QtCore.Qt.LeftButton)
    assert w._display_mode == "wiggle"
    assert w.imageItem_seismic.image is None
    qtbot.mouseClick(w.radio_density, QtCore.Qt.LeftButton)
    assert w._display_mode == "density"
