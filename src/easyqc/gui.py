import sys  # We need sys so that we can pass argv to QApplication
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtGui import QTransform, QIcon

import pyqtgraph as pg

import easyqc.qt


class EasyQC(QtWidgets.QMainWindow):
    """
    This is the view in the MVC approach
    """
    layers = None  # used for additional scatter layers

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
        # wave by Diana Militano from the Noun Project
        self.layers = {}
        self.ctrl = Controller(self)
        uic.loadUi(Path(__file__).parent.joinpath('easyqc.ui'), self)
        self.setWindowIcon(QIcon(str(Path(__file__).parent.joinpath('easyqc.svg'))))
        background_color = self.palette().color(self.backgroundRole())
        # init the seismic density display
        self.plotItem_seismic.setAspectLocked(False)
        self.imageItem_seismic = pg.ImageItem()
        self.plotItem_seismic.setBackground(background_color)
        self.plotItem_seismic.addItem(self.imageItem_seismic)
        self.viewBox_seismic = self.plotItem_seismic.getPlotItem().getViewBox()
        # init the header display and link X and Y axis with density display
        self.plotDataItem_header_h = pg.PlotDataItem()
        self.plotItem_header_h.addItem(self.plotDataItem_header_h)
        self.plotItem_seismic.setXLink(self.plotItem_header_h)
        self.plotDataItem_header_v = pg.PlotDataItem()
        self.plotItem_header_h.setBackground(background_color)
        self.plotItem_header_v.addItem(self.plotDataItem_header_v)
        self.plotItem_header_v.setBackground(background_color)
        self.plotItem_seismic.setYLink(self.plotItem_header_v)
        # set the ticks so that they don't auto scale and ruin the axes link
        ax = self.plotItem_seismic.getAxis('left')
        ax.setStyle(tickTextWidth=60, autoReduceTextSpace=False, autoExpandTextSpace=False)
        ax = self.plotItem_header_h.getAxis('left')
        ax.setStyle(tickTextWidth=60, autoReduceTextSpace=False, autoExpandTextSpace=False)
        ax = self.plotItem_header_v.getAxis('left')
        ax.setStyle(showValues=False)
        # connect signals and slots

        s = self.viewBox_seismic.scene()
        # vb.scene().sigMouseMoved.connect(self.mouseMoveEvent)
        self.proxy = pg.SignalProxy(s.sigMouseMoved, rateLimit=60, slot=self.mouseMoveEvent)
        s.sigMouseClicked.connect(self.mouseClick)
        self.lineEdit_gain.returnPressed.connect(self.editGain)
        self.lineEdit_sort.returnPressed.connect(self.editSort)
        self.comboBox_header.activated[str].connect(self.ctrl.set_header)
        self.viewBox_seismic.sigRangeChanged.connect(self.on_sigRangeChanged)
        self.horizontalScrollBar.sliderMoved.connect(self.on_horizontalSliderChange)
        self.verticalScrollBar.sliderMoved.connect(self.on_verticalSliderChange)

    """
    View Methods
    """
    def closeEvent(self, event):
        self.destroy()

    def keyPressEvent(self, e):
        """
        page-up / ctrl + a :  gain up
        page-down / ctrl + z : gain down
        ctrl + p : propagate display to current windows
        up/down/left/right arrows: pan using keys
        :param e:
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
            self.translate_seismic(k, m == QtCore.Qt.ControlModifier)

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
        c, t, a, h = self.ctrl.cursor2timetraceamp(qpoint)
        self.label_x.setText(f"{c:.0f}")
        self.label_t.setText(f"{t:.4f}")
        self.label_amp.setText(f"{a:2.2E}")
        self.label_h.setText(f"{h:.4f}")

    def translate_seismic(self, k, cm):
        """
        Resizes vertical or horizontal on a KeyPress
        :param k:  translate by 1./7
        :param cm (bool): if the control modifier has been pressed, translate by 1./2
        :return:
        """
        r = self.viewBox_seismic.viewRect()
        xlim, ylim = self.ctrl.limits()
        FAC = 1 / 2 if cm else 1 / 7
        dy = FAC * r.height()
        dx = FAC * r.width()
        if k == QtCore.Qt.Key_Down:
            yr = np.array([r.y(), r.y() + r.height()]) + dy
            yr += np.min([0, ylim[1] - yr[1]])
            self.viewBox_seismic.setYRange(yr[0], yr[1], padding=0)
        elif k == QtCore.Qt.Key_Left:
            xr = np.array([r.x(), r.x() + r.width()]) - dx
            xr += np.max([0, xlim[0] - xr[0]])
            self.viewBox_seismic.setXRange(xr[0], xr[1], padding=0)
        elif k == QtCore.Qt.Key_Right:
            xr = np.array([r.x(), r.x() + r.width()]) + dx
            xr += np.min([0, xlim[1] - xr[1]])
            self.viewBox_seismic.setXRange(xr[0], xr[1], padding=0)
        elif k == QtCore.Qt.Key_Up:
            yr = np.array([r.y(), r.y() + r.height()]) - dy
            yr += np.max([0, ylim[0] - yr[0]])
            self.viewBox_seismic.setYRange(yr[0], yr[1], padding=0)

    def on_sigRangeChanged(self, r):

        def set_scroll(sb, r, l):
            # sb: scroll bar object, r: current range, l: axes limits
            # cf. https://doc.qt.io/qt-5/qscrollbar.html
            range = (r[1] - r[0])
            doclength = (l[1] - l[0])
            maximum = int((doclength - range) / doclength * 65536)
            sb.setMaximum(maximum)
            sb.setPageStep(65536 - maximum)
            sb.setValue(int((r[0] - l[0]) / doclength * 65536))

        xr, yr = self.viewBox_seismic.viewRange()
        xl, yl = self.ctrl.limits()
        set_scroll(self.horizontalScrollBar, xr, xl)
        set_scroll(self.verticalScrollBar, yr, yl)

    def on_horizontalSliderChange(self, r):
        l = self.ctrl.limits()[0]
        r = self.viewBox_seismic.viewRange()[0]
        x = float(self.horizontalScrollBar.value()) / 65536 * (l[1] - l[0]) + l[0]
        self.viewBox_seismic.setXRange(x, x + r[1] - r[0], padding=0)

    def on_verticalSliderChange(self, r):
        l = self.ctrl.limits()[1]
        r = self.viewBox_seismic.viewRange()[1]
        y = float(self.verticalScrollBar.value()) / 65536 * (l[1] - l[0]) + l[0]
        self.viewBox_seismic.setYRange(y, y + r[1] - r[0], padding=0)


class Controller:

    def __init__(self, view):
        self.view = view
        self.model = Model(None, None)
        self.order = None
        self.transform = None  # affine transform image indices 2 data domain
        self.gain = None
        self.trace_indices = None
        self.hkey = None

    def remove_all_layers(self):
        layers_dict = self.view.layers.copy()
        for label in layers_dict:
            self.remove_layer_from_label(label)

    def remove_layer_from_label(self, label):
        current_layer = self.view.layers.get(label)
        if current_layer is not None:
            current_layer['layer'].clear()
            self.view.plotItem_seismic.removeItem(current_layer['layer'])
            self.view.layers.pop(label)

    def add_scatter(self, x, y, rgb=None, label='default'):
        """
        Adds a scatter layer to the display (removing any previous one if any)
        """
        rgb = rgb or (0, 255, 0)
        self.remove_layer_from_label(label)
        new_scatter = pg.ScatterPlotItem()
        self.view.layers[label] = {'layer': new_scatter, 'type': 'scatter'}
        self.view.plotItem_seismic.addItem(new_scatter)
        new_scatter.setData(x=x, y=y, brush=pg.mkBrush(rgb), name=label)

    def cursor2timetraceamp(self, qpoint):
        """Used for the mouse hover function over seismic display, returns trace, time,
          amplitude,and header"""
        ixy = self.cursor2ind(qpoint)
        a = self.model.data[ixy[0], ixy[1]]
        xy_ = np.matmul(self.transform, np.array([ixy[0], ixy[1], 1]))
        t = xy_[self.model.taxis]
        c = xy_[self.model.caxis]
        h = self.model.header[self.hkey][ixy[self.model.caxis]]
        return c, t, a, h

    def cursor2ind(self, qpoint):
        """ image coordinates over the seismic display"""
        ix = np.max((0, np.min((int(np.floor(qpoint.x())), self.model.nx - 1))))
        iy = np.max((0, np.min((int(np.round(qpoint.y())), self.model.ny - 1))))
        return ix, iy

    def limits(self):
        # returns the xlims and ylims of the data in the data space (time, trace)
        ixlim = [0, self.model.nx]
        iylim = [0, self.model.ny]
        x, y, _ = np.matmul(self.transform, np.c_[ixlim, iylim, [1, 1]].T)
        return x, y

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
        traces = np.arange(self.trace_indices.size)
        values = self.model.header[self.hkey][self.trace_indices]
        if self.model.taxis == 1:
            self.view.plotDataItem_header_h.setData(x=traces, y=values)
        elif self.model.taxis == 0:
            self.view.plotDataItem_header_v.setData(y=traces, x=values)

    def sort(self, keys):
        if not(set(keys).issubset(set(self.model.header.keys()))):
            print("Wrong input")
            return
        elif len(keys) == 0:
            return
        self.trace_indices = np.lexsort([self.model.header[k] for k in keys])
        self.redraw()

    def update_data(self, data, h=None, si=.002, gain=None, x0=0, t0=0, taxis=1):
        """
        data is a 2d array [ntr, nsamples]
        if 3d the first dimensions are merged in ntr and the last is nsamples
        update_data(self, data=None, h=0.002, gain=None)
        """
        # reshape a 3d+ array in 2d to plot as an image
        self.remove_all_layers()
        if data.ndim >= 3:
            data = np.reshape(data, (-1, data.shape[-1]))
        self.model.set_data(data, si=si, header=h, x0=x0, t0=t0, taxis=taxis)
        self.gain = gain or self.model.auto_gain()
        self.trace_indices = np.arange(self.model.ntr)  # this will contain selection and sort
        clim = [x0 - .5, x0 + self.model.ntr - .5]
        tlim = [t0, t0 + self.model.ns * self.model.si]
        if taxis == 0:  # time is the 0 dimension and the horizontal axis
            xlim, ylim = (tlim, clim)
            transform = [si, 0., 0., 0., 1, 0., t0 - si / 2, x0 - .5, 1.]
            self.view.imageItem_seismic.setImage(data[:, self.trace_indices])
        elif taxis == 1:  # time is the 1 dimension and vertical axis
            xlim, ylim = (clim, tlim)
            transform = [1., 0., 0., 0., si, 0., x0 - .5, t0 - si / 2, 1.]
            self.view.imageItem_seismic.setImage(data[self.trace_indices, :])
            self.view.plotItem_seismic.invertY()
        else:
            ValueError('taxis must be 0 (horizontal axis) or 1 (vertical axis)')
        self.transform = np.array(transform).reshape((3, 3)).T
        self.view.imageItem_seismic.setTransform(QTransform(*transform))
        self.view.plotItem_header_h.setLimits(xMin=xlim[0], xMax=xlim[1])
        self.view.plotItem_header_v.setLimits(yMin=ylim[0], yMax=ylim[1])
        self.view.plotItem_seismic.setLimits(xMin=xlim[0], xMax=xlim[1], yMin=ylim[0], yMax=ylim[1])
        # reset the view
        xlim, ylim = self.limits()
        self.view.viewBox_seismic.setXRange(*xlim, padding=0)
        self.view.viewBox_seismic.setYRange(*ylim, padding=0)
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

    def set_data(self, data, header=None, si=None, t0=0, x0=0, taxis=1):
        assert header or si
        # intrinsic data
        self.x0 = x0
        self.t0 = t0
        self.header = header
        self.data = data
        self.taxis = taxis
        self.nx, self.ny = self.data.shape
        if self.taxis == 1:
            self.ntr, self.ns = self.data.shape
            self.caxis = 0
        else:
            self.ns, self.ntr = self.data.shape
            self.caxis = 1
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
        rmsnan = np.nansum(self.data ** 2, axis=self.taxis) / np.sum(
            ~np.isnan(self.data), axis=self.taxis)
        return 20 * np.log10(np.median(np.sqrt(rmsnan)))


def viewseis(w=None, si=.002, h=None, title=None, t0=0, x0=0, taxis=1):
    """
    viewseis(w, h, 'processed')
    :param w: 2D array (ntraces, nsamples)
    :param h: sample rate if float, dictionary (si)
    :param t0:
    :param x0:
    :param title: Tag for the window.
    :return: EasyQC object
    """
    easyqc.qt.create_app()
    eqc = EasyQC._get_or_create(title=title)
    if w is not None:
        eqc.ctrl.update_data(w, h=h, si=si, t0=t0, x0=x0, taxis=taxis)
    eqc.show()
    return eqc


if __name__ == '__main__':
    eqc = viewseis(None)
    app = pg.Qt.mkQApp()
    sys.exit(app.exec_())
