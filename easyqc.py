import sys  # We need sys so that we can pass argv to QApplication
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtGui import QTransform

import pyqtgraph as pg

import qt


class EasyQC(QtWidgets.QMainWindow):

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, EasyQC)]

    @staticmethod
    def _get_or_create(title=None):
        eqc = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                          EasyQC._instances()), None)
        if eqc is None:
            eqc = EasyQC()
            eqc.setWindowTitle(title)
        return eqc

    def __init__(self, *args, **kwargs):
        super(EasyQC, self).__init__(*args, **kwargs)
        self.ctrl = Controller(self)
        uic.loadUi(Path(__file__).parent.joinpath('easyqc.ui'), self)
        # init the seismic density display
        self.plotItem_seismic.setAspectLocked(False)
        self.plotItem_seismic.invertY()
        self.imageItem_seismic = pg.ImageItem()
        self.plotItem_seismic.addItem(self.imageItem_seismic)
        # init the header display and link X-axis with density
        self.plotDataItem_header = pg.PlotDataItem()
        self.plotItem_Header.addItem(self.plotDataItem_header)
        self.plotItem_seismic.setXLink(self.plotItem_Header)
        # connect signals and slots
        s = self.plotItem_seismic.getViewBox().scene()
        # vb.scene().sigMouseMoved.connect(self.mouseMoveEvent)
        self.proxy = pg.SignalProxy(s.sigMouseMoved, rateLimit=60, slot=self.mouseMoveEvent)
        s.sigMouseClicked.connect(self.mouseClick)
        self.lineEdit_gain.returnPressed.connect(self.editGain)
        self.lineEdit_sort.returnPressed.connect(self.editSort)
        self.comboBox_header.activated[str].connect(self.ctrl.set_header)
    """
    View Methods
    """
    def closeEvent(self, event):
        self.destroy()

    def keyPressEvent(self, e):
        """
        page-up / ctrl + a :  gain up
        page-down / ctrl + z : gain down
        ctrl + p : propagate display
        :param e:
        :return:
        """
        k, m = (e.key(), e.modifiers())
        if k == QtCore.Qt.Key_PageUp or (  # page up / ctrl + a
                m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_A):
            self.ctrl.set_gain(self.ctrl.gain - 3)
        elif k == QtCore.Qt.Key_PageDown or (  # page down / ctrl + z
                m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_Z):
            self.ctrl.set_gain(self.ctrl.gain + 3)
        elif m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_P:
            self.ctrl.propagate()
        elif k in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_Down):
            self.translate_seismic(k)

    def editGain(self):
        try:
            gain = float(self.lineEdit_gain.text())
        except ValueError():
            pass  # tddo set colour self.lineEdit_gain
        self.ctrl.set_gain(gain)

    def editSort(self):
        keys = self.lineEdit_sort.text().split(' ')
        self.ctrl.sort(keys)

    def mouseClick(self, event):
        if not event.double():
            return
        qxy = self.imageItem_seismic.mapFromScene(event.scenePos())
        tr, s = (qxy.x(), qxy.y())
        print(tr, s)

    def mouseMoveEvent(self, scenepos):
        if isinstance(scenepos, tuple):
            scenepos = scenepos[0]
        else:
            return
        qpoint = self.imageItem_seismic.mapFromScene(scenepos)
        ix, _, a, t, h = self.ctrl.cursor2timetraceamp(qpoint)
        self.label_x.setText(f"{ix:.0f}")
        self.label_t.setText(f"{t:.4f}")
        self.label_amp.setText(f"{a:2.2E}")
        self.label_h.setText(f"{h:.4f}")

    def translate_seismic(self, k):
        pass
        # if k == QtCore.Qt.Key_Up:
        #     rx, ry = (0., - .75)
        # elif k == QtCore.Qt.Key_Left:
        #     rx, ry = (- .75, 0.)
        # elif k == QtCore.Qt.Key_Right:
        #     rx, ry = (.75, 0.)
        # elif k == QtCore.Qt.Key_Down:
        #     rx, ry = (0., .75)
        # r = self.plotItem_seismic.viewRect()
        # print(rx, ry)
        # r.translate(r.width() * rx, r.height() * ry)
        # print(r)
        # self.plotItem_seismic.setRange(self.plotItem_seismic.range)


class Controller:

    def __init__(self, view):
        self.view = view
        self.model = Model(None, None)
        self.order = None
        self.transform = None  # affine transform image indices 2 data domain
        self.gain = None
        self.trace_indices = None
        self.hkey = None

    def cursor2timetraceamp(self, qpoint):
        """Used for the mouse hover function over seismic display"""
        ix, iy = self.cursor2ind(qpoint)
        a = self.model.data[ix, iy]
        x, t, _ = np.matmul(self.transform, np.array([ix, iy, 1]))
        h = self.model.header[self.hkey][ix]
        return ix, iy, a, t, h

    def cursor2ind(self, qpoint):
        """ image coordinates over the seismic display"""
        ix = np.max((0, np.min((int(np.floor(qpoint.x())), self.model.ntr - 1))))
        iy = np.max((0, np.min((int(np.round(qpoint.y())), self.model.ns - 1))))
        return ix, iy

    def propagate(self):
        """ set all the eqc instances at the same position/gain scales for flip comparisons """
        eqcs = self.view._instances()
        for eqc in eqcs:
            if eqc is self.view:
                continue
            else:
                eqc.setGeometry(self.view.geometry())
                eqc.ctrl.set_gain(self.gain)
                eqc.plotItem_seismic.setXLink(self.view.plotItem_seismic)
                eqc.plotItem_seismic.setYLink(self.view.plotItem_seismic)
                # also propagate sorting
                eqc.lineEdit_sort.setText(self.view.lineEdit_sort.text())
                eqc.ctrl.sort(eqc.lineEdit_sort.text())

    def redraw(self):
        """ redraw seismic and headers with order and selection"""
        self.view.imageItem_seismic.setImage(self.model.data[self.trace_indices, :])
        self.set_header()
        self.set_gain()

    def set_gain(self, gain=None):
        if gain is None:
            gain = self.gain
        else:
            self.gain = gain
        levels = 10 ** (self.gain / 20) * 4 * np.array([-1, 1])
        self.view.imageItem_seismic.setLevels(levels)
        self.view.lineEdit_gain.setText(f"{gain:.1f}")

    def set_header(self):
        key = self.view.comboBox_header.currentText()
        if key not in self.model.header.keys():
            return
        self.hkey = key
        self.view.plotDataItem_header.setData(
            x=np.arange(self.trace_indices.size),
            y=self.model.header[self.hkey][self.trace_indices])

    def sort(self, keys):
        if not(set(keys).issubset(set(self.model.header.keys()))):
            print("Wrong input")
            return
        elif len(keys) == 0:
            return
        self.trace_indices = np.lexsort([self.model.header[k] for k in keys])
        self.redraw()

    def update_data(self, data, h=None, si=.002, gain=None, x0=0, t0=0):
        """
        data is a 2d array [ntr, nsamples]
        if 3d the first dimensions are merged in ntr and the last is nsamples
        update_data(self, data=None, h=0.002, gain=None)
        """
        # reshape a 3d+ array in 2d to plot as an image
        if data.ndim >= 3:
            data = np.reshape(data, (-1, data.shape[-1]))
        self.model.set_data(data, si=si, header=h, x0=x0, t0=t0)
        self.gain = gain or self.model.auto_gain()
        self.trace_indices = np.arange(self.model.ntr)  # this will contain selection and sort
        self.view.imageItem_seismic.setImage(data[self.trace_indices, :])
        transform = [1., 0., 0., 0., si, 0., x0 - .5, t0 - si / 2, 1.]
        self.transform = np.array(transform).reshape((3, 3)).T
        self.view.imageItem_seismic.setTransform(QTransform(*transform))
        self.view.plotItem_seismic.setLimits(xMin=x0 - .5, xMax=x0 + data.shape[0] - .5,
                                             yMin=t0, yMax=t0 + data.shape[1] * self.model.si)
        # set the header combo box keys
        if isinstance(self.model.header, dict):
            self.view.comboBox_header.clear()
            for hname in self.model.header.keys():
                self.view.comboBox_header.addItem(hname)
        self.set_gain(gain=gain)
        self.set_header()


@dataclass
class Model:
    """Class for keeping track of the visualized data"""
    data: np.array
    header: np.array
    si: float = 1.

    def set_data(self, data, header=None, si=None, t0=0, x0=0):
        assert header or si
        # intrinsic data
        self.x0 = x0
        self.t0 = t0
        self.header = header
        self.data = data
        self.ntr, self.ns = self.data.shape
        # get the sampling reate
        if si is not None:
            self.si = si
        else:
            if isinstance(header, float):
                self.si = header
            else:
                self.si = header.si[0]
        self.header = {'trace': np.arange(self.ntr)}
        if header is not None:
            self.header.update(header)

    def auto_gain(self) -> float:
        return 20 * np.log10(np.median(np.sqrt(
            np.nansum(self.data ** 2, axis=1) / np.sum(~np.isnan(self.data), axis=1))))


def viewdata(w=None, si=.002, h=None, title=None, t0=0, x0=0):
    """
    viewdata(w, h, 'processed')
    :param w: 2D array (ntraces, nsamples)
    :param h: sample rate if float, dictionary (si)
    :param t0:
    :param x0:
    :param title: Tag for the window.
    :return: EasyQC object
    """
    qt.create_app()
    eqc = EasyQC._get_or_create(title=title)
    if w is not None:
        eqc.ctrl.update_data(w, h=h, si=si, t0=t0, x0=x0)
    eqc.show()
    return eqc


if __name__ == '__main__':
    eqc = viewdata(None)
    app = pg.Qt.mkQApp()
    sys.exit(app.exec_())
