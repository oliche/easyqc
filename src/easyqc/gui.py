import sys  # We need sys so that we can pass argv to QApplication
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, uic

import easyqc.qt as qt
import pyqtgraph as pg
from easyqc.pgtools import ImShowSpectrogram

PARAMS_TRACE_PLOTS = {
    'neighbors': 2,
    'color': pg.mkColor((31, 119, 180)),
}


class EasyQC(QtWidgets.QMainWindow):
    """
    This is the view in the MVC approach
    """
    layers = None  # used for additional scatter layers
    QT_APP = None

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = qt.create_app()
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
        self.setWindowIcon(QtGui.QIcon(str(Path(__file__).parent.joinpath('easyqc.svg'))))
        background_color = self.palette().color(self.backgroundRole())
        # init the seismic density display
        self.plotItem_seismic.setAspectLocked(False)
        self.imageItem_seismic = pg.ImageItem()
        self.plotItem_seismic.setBackground(background_color)
        self.plotItem_seismic.addItem(self.imageItem_seismic)
        self.viewBox_seismic = self.plotItem_seismic.getPlotItem().getViewBox()
        self._init_menu()
        self._init_cmenu()
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
        # prepare placeholders for hover windows
        self.hoverPlotWidgets = {'Trace': None, 'Spectrum': None, 'Spectrogram': None}
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

    def _init_menu(self):
        # pre-defined colormaps
        self.actionColormap_CET_D6.triggered.connect(lambda: self.setColorMap('CET-D6'))
        self.actionColormap_CET_D1.triggered.connect(lambda: self.setColorMap('CET-D1'))
        self.actionColormap_CET_L2.triggered.connect(lambda: self.setColorMap('CET-L2'))
        self.actionColormap_MPL_PuOr.triggered.connect(lambda: self.setColorMap('PuOr'))

    def _init_cmenu(self):
        """
        Setup context menus - on instantiation only
        """
        self.viewBox_seismic.scene().contextMenu = None  # this gets rid of the export context menu
        self.plotItem_seismic.plotItem.ctrlMenu = None  # this gets rid of the plot context menu
        for act in self.viewBox_seismic.menu.actions():
            if act.text() == 'View All':
                continue
            self.viewBox_seismic.menu.removeAction(act)
        # and add ours
        self.viewBox_seismic.menu.addSeparator()
        act = QtWidgets.QAction("View Trace", self.viewBox_seismic.menu)
        act.triggered.connect(self.cmenu_ViewTrace)
        self.viewBox_seismic.menu.addAction(act)
        act = QtWidgets.QAction("View Spectrum", self.viewBox_seismic.menu)
        act.triggered.connect(self.cmenu_ViewSpectrum)
        self.viewBox_seismic.menu.addAction(act)
        act = QtWidgets.QAction("View Spectrogram", self.viewBox_seismic.menu)
        act.triggered.connect(self.cmenu_ViewSpectrogram)
        self.viewBox_seismic.menu.addAction(act)

    """
    View Methods
    """
    def keyPressEvent(self, e):
        """
        page-up / ctrl + a :  gain up
        page-down / ctrl + z : gain down
        ctrl + p : propagate display to current windows
        up/down/left/right arrows: pan using keys
        :param e:
        """
        k, m = (e.key(), e.modifiers())
        # page up / ctrl + a
        if k == QtCore.Qt.Key_PageUp or (
                m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_A):
            self.ctrl.set_gain(self.ctrl.gain - 3)
        # page down / ctrl + z
        elif k == QtCore.Qt.Key_PageDown or (
                m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_Z):
            self.ctrl.set_gain(self.ctrl.gain + 3)
        # control + P: propagate
        elif m == QtCore.Qt.ShiftModifier and k == QtCore.Qt.Key_P:
            self.ctrl.propagate(explode=True)
        elif m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_P:
            self.ctrl.propagate()
        # arrows keys move seismic
        elif k in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_Down):
            self.translate_seismic(k, m == QtCore.Qt.ControlModifier)
        # ctrl + s: screenshot to clipboard
        elif m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_S:
            qtapp = QtWidgets.QApplication.instance()
            qtapp.clipboard().setPixmap(self.plotItem_seismic.grab())

    def editGain(self):
        self.ctrl.set_gain()

    def editSort(self, redraw=True):
        keys = self.lineEdit_sort.text().split(' ')
        self.ctrl.sort(keys, redraw=redraw)

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
        htxt = h if isinstance(h, str) else f"{h:.4f}"
        self.label_h.setText(htxt)
        for key in self.hoverPlotWidgets:
            if self.hoverPlotWidgets[key] is not None and self.hoverPlotWidgets[key].isVisible():
                self.ctrl.update_hover(qpoint, key)

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

        def set_scroll(sb, r, b):
            # sb: scroll bar object, r: current range, b: axes limits (bounds)
            # cf. https://doc.qt.io/qt-5/qscrollbar.html
            range = (r[1] - r[0])
            doclength = (b[1] - b[0])
            maximum = int((doclength - range) / doclength * 65536)
            sb.setMaximum(maximum)
            sb.setPageStep(65536 - maximum)
            sb.setValue(int((r[0] - b[0]) / doclength * 65536))

        xr, yr = self.viewBox_seismic.viewRange()
        xl, yl = self.ctrl.limits()
        set_scroll(self.horizontalScrollBar, xr, xl)
        set_scroll(self.verticalScrollBar, yr, yl)

    def on_horizontalSliderChange(self, r):
        b = self.ctrl.limits()[0]
        r = self.viewBox_seismic.viewRange()[0]
        x = float(self.horizontalScrollBar.value()) / 65536 * (b[1] - b[0]) + b[0]
        self.viewBox_seismic.setXRange(x, x + r[1] - r[0], padding=0)

    def on_verticalSliderChange(self, r):
        b = self.ctrl.limits()[1]
        r = self.viewBox_seismic.viewRange()[1]
        y = float(self.verticalScrollBar.value()) / 65536 * (b[1] - b[0]) + b[0]
        self.viewBox_seismic.setYRange(y, y + r[1] - r[0], padding=0)

    def _cmenu_hover(self, key, image=False):
        """Creates the plot widget for a given key: could be 'Trace', 'Spectrum', or 'Spectrogram'"""
        if self.hoverPlotWidgets[key] is None:
            if image and key == 'Spectrogram':
                self.hoverPlotWidgets[key] = ImShowSpectrogram()
            else:
                self.hoverPlotWidgets[key] = pg.plot([0], [0], pen=pg.mkPen(color=[180, 180, 180]), connect="finite")
                self.hoverPlotWidgets[key].addItem(
                    pg.PlotCurveItem(
                        [0], [0], pen=pg.mkPen(color=PARAMS_TRACE_PLOTS['color'], width=1), connect="finite"))
                self.hoverPlotWidgets[key].addItem(
                    pg.PlotDataItem([0], [0], symbolPen=pg.mkPen(color=[255, 0, 0]), symbolSize=7, symbol='star'))
                self.hoverPlotWidgets[key].setBackground(pg.mkColor('#ffffff'))
        self.hoverPlotWidgets[key].setVisible(True)

    def cmenu_ViewTrace(self):
        self._cmenu_hover('Trace')

    def cmenu_ViewSpectrum(self):
        self._cmenu_hover('Spectrum')

    def cmenu_ViewSpectrogram(self):
        self._cmenu_hover('Spectrogram', image=True)

    def setColorMap(self, cmap):
        """
        Set the colormap for the seismic display - useful for the propagation feature
        :param cmap:
        :return:
        """
        if isinstance(cmap, str) and cmap not in pg.colormap.listMaps():
            cmap = pg.colormap.getFromMatplotlib(cmap)
        self.imageItem_seismic.setColorMap(cmap)


class Controller:

    def __init__(self, view):
        self.view = view
        self.model = Model(None, None)
        self.order = None
        self.transform = np.eye(3)  # affine transform image indices 2 data domain
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

    def _add_plotitem(self, plot_item_class, x, y, rgb=None, label='default', brush=None, pen=None, **kwargs):
        """
        Generic way to add a plot item to the main seismic view
        """
        rgb = (0, 255, 0) if rgb is None else rgb
        self.remove_layer_from_label(label)
        new_scatter = plot_item_class()
        self.view.layers[label] = {'layer': new_scatter, 'type': 'scatter'}
        self.view.plotItem_seismic.addItem(new_scatter)
        brush = pg.mkBrush(rgb) if brush is None else brush
        pen = pg.mkPen(rgb) if pen is None else pen
        new_scatter.setData(x=x, y=y, brush=brush, name=label, pen=pen, **kwargs)
        return new_scatter

    def add_curve(self, *args, **kwargs):
        """
        Adds a curve layer to the display (removing any previous one with the same name if any)
        """
        return self._add_plotitem(pg.PlotCurveItem, *args, **kwargs)

    def add_scatter(self, *args, **kwargs):
        """
        Adds a sca layer to the display (removing any previous one with the same name if any)
        """
        return self._add_plotitem(pg.ScatterPlotItem, *args, **kwargs)

    def cursor2timetraceamp(self, qpoint):
        """Used for the mouse hover function over seismic display, returns trace, time,
          amplitude,and header"""
        ixy = self.cursor2ind(qpoint)
        a = self.model.data[ixy[0], ixy[1]]
        xy_ = np.matmul(self.transform, np.array([qpoint.x(), qpoint.y(), 1]))
        t = xy_[self.model.taxis]
        c = ixy[self.model.caxis]
        h = self.model.header[self.hkey][ixy[self.model.caxis]]
        return c, t, a, h

    def cursor2ind(self, qpoint):
        """ image coordinates over the seismic display"""
        ix = int(np.maximum(0, np.minimum(qpoint.x() - self.transform[1, 2], self.model.nx - 1)))
        iy = int(np.maximum(0, np.minimum(qpoint.y() - self.transform[1, 1], self.model.ny - 1)))
        return ix, iy

    def limits(self):
        # returns the xlims and ylims of the data in the data space (time, trace)
        ixlim = [0, self.model.nx]
        iylim = [0, self.model.ny]
        x, y, _ = np.matmul(self.transform, np.c_[ixlim, iylim, [1, 1]].T)
        return x, y

    def propagate(self, explode=False):
        """
        set all the eqc instances at the same position/gain scales for flip comparisons
        If explodes is set to True, splits windows in groups of two side by side
        """
        eqcs = self.view._instances()
        if len(eqcs) == 1:
            return
        for i, eqc in enumerate(eqcs):
            if eqc is self.view:
                continue
            else:
                eqc.setColorMap(self.view.imageItem_seismic.getColorMap() or 'CET-L2')
                eqc.setGeometry(self.view.geometry())
                eqc.ctrl.set_gain(self.gain)
                eqc.plotItem_seismic.setXLink(self.view.plotItem_seismic)
                eqc.plotItem_seismic.setYLink(self.view.plotItem_seismic)
                # eqc.plotItem_seismic.setXLink(eqc.plotItem_header_h)
                # eqc.plotItem_seismic.setYLink(eqc.plotItem_header_v)
                # also propagate sorting
                eqc.lineEdit_sort.setText(self.view.lineEdit_sort.text())
                eqc.ctrl.sort(eqc.lineEdit_sort.text())
            # every odd figure is shifted alongside the first one
            if explode and i % 2 == 1:
                rect = self.view.geometry()
                if i % 2 == 1:
                    rect.translate(rect.width(), 0)
                eqc.setGeometry(rect)

    def redraw(self):
        """ redraw seismic and headers with order and selection"""
        # np.take could look neater but it's actually much slower than straight indexing
        if self.model.taxis == 1:
            self.view.imageItem_seismic.setImage(self.model.data[self.trace_indices, :])
        elif self.model.taxis == 0:
            self.view.imageItem_seismic.setImage(self.model.data[:, self.trace_indices])
        self.set_header()
        self.set_gain()

    def set_gain(self, gain=None):
        if gain is None:
            gain = self.gain
        levels = 10 ** (gain / 20) * 4 * np.array([-1, 1])
        self.view.imageItem_seismic.setLevels(levels)
        self.view.lineEdit_gain.setText(f"{gain:.1f}")

    @property
    def gain(self):
        return float(self.view.lineEdit_gain.text()) or self.model.auto_gain()

    def set_header(self):
        key = self.view.comboBox_header.currentText()
        if key not in self.model.header.keys():
            return
        self.hkey = key
        traces = np.arange(self.trace_indices.size)
        values = self.model.header[self.hkey][self.trace_indices]
        # skip the plotting part for non-numeric arrays
        if not np.issubdtype(values.dtype, np.number):
            return
        if self.model.taxis == 1:
            self.view.plotDataItem_header_h.setData(x=traces, y=values)
        elif self.model.taxis == 0:
            self.view.plotDataItem_header_v.setData(y=traces, x=values)

    def snapshot(self, file, xrange=None, yrange=None, gain=None, window_size=None):
        """
        Saves a snapshot of the current view to a file
        :param file: str or pathlib.Path
        :param xrange: range of the horizontal axis in data space
        :param yrange: range of the vertical axis in data space
        :param gain: gain of the raster in dB
        :param window_size: tuple of width and height in pixels
        :return:
        """
        if yrange is not None:
            self.view.viewBox_seismic.setYRange(*yrange)
        if xrange is not None:
            self.view.viewBox_seismic.setXRange(*xrange)
        if gain is not None:
            self.set_gain(gain)
        if window_size is not None:
            self.view.resize(*window_size)
        self.view.grab().save(str(file))

    def sort(self, keys, redraw=True):
        if not (set(keys).issubset(set(self.model.header.keys()))):
            return
        elif len(keys) == 0:
            return
        self.trace_indices = np.lexsort([self.model.header[k] for k in keys])
        if redraw:
            self.redraw()

    def update_data(self, data, h=None, si=.002, gain=None, x0=0, t0=0, taxis=1):
        """
        data is a 2d array [ntr, nsamples]
        if 3d the first dimensions are merged in ntr and the last is nsamples
        update_data(self, data=None, h=0.002, gain=None)
        """
        # reshape a 3d+ array in 2d to plot as an image
        self.remove_all_layers()
        # if the data has the same shape as the current model data, keep axis all the same
        update_axis = self.model.data is None or self.model.data.shape != data.shape
        if data.ndim >= 3:
            data = np.reshape(data, (-1, data.shape[-1]))
        self.model.set_data(data, si=si, header=h, x0=x0, t0=t0, taxis=taxis)
        self.trace_indices = np.arange(self.model.ntr)
        self.view.editSort(redraw=False) # this sets the self.trace_indices according to sort order
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
        self.view.imageItem_seismic.setTransform(QtGui.QTransform(*transform))
        self.view.plotItem_header_h.setLimits(xMin=xlim[0], xMax=xlim[1])
        self.view.plotItem_header_v.setLimits(yMin=ylim[0], yMax=ylim[1])
        self.view.plotItem_seismic.setLimits(xMin=xlim[0], xMax=xlim[1], yMin=ylim[0], yMax=ylim[1])
        # reset the view
        if update_axis:
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

    def update_hover(self, qpoint, key):
        c, t, a, _ = self.cursor2timetraceamp(qpoint)
        if key == 'Trace':
            plotitem = self.view.hoverPlotWidgets[key].getPlotItem()
            traces = self.model.get_trace(c, neighbors=PARAMS_TRACE_PLOTS['neighbors'])
            nc = traces.shape[1]
            xdata = np.tile(np.r_[self.model.tscale, np.nan], nc).T
            ydata = np.r_[traces, np.ones((1, nc)) * np.nan].T
            plotitem.items[0].setData(xdata.flatten(), ydata.flatten())
            trace = self.model.get_trace(c)
            plotitem.items[1].setData(self.model.tscale, trace)
            plotitem.items[2].setData([t], [a])
            plotitem.setXRange(*self.trange)
        elif key == 'Spectrum':
            plotitem = self.view.hoverPlotWidgets[key].getPlotItem()
            plotitem.items[0].setData(*self.model.get_trace_spectrum(c, trange=self.trange))
        elif key == 'Spectrogram':
            self.view.hoverPlotWidgets[key].set_data(self.model.get_trace(c), fs=1 / self.model.si)

    @property
    def trange(self):
        """
        returns the current time range of the view
        :return: 2 floats list
        """
        return self.view.viewBox_seismic.viewRange()[self.model.taxis]

    @property
    def crange(self):
        """
        returns the current channel range of the view
        :return: 2 floats list
        """
        return self.view.viewBox_seismic.viewRange()[self.model.caxis]


@dataclass
class Model:
    """Class for keeping track of the visualized data"""
    data: np.array
    header: np.array
    si: float = 1.
    nx: int = 1
    ny: int = 1

    def auto_gain(self) -> float:
        rmsnan = np.nansum(self.data ** 2, axis=self.taxis) / np.sum(
            ~np.isnan(self.data), axis=self.taxis)
        return 20 * np.log10(np.median(np.sqrt(rmsnan)))

    def get_trace_spectrogram(self, c, trange=None):
        from scipy.signal import spectrogram
        tr = self.get_trace(c, trange=trange)
        fscale, tscale, tf = spectrogram(tr, fs=1 / self.si, nperseg=50, nfft=512, window='cosine', noverlap=48)
        tscale += trange[0]
        tf = 20 * np.log10(tf + np.finfo(float).eps)
        return fscale, tscale, tf

    def get_trace_spectrum(self, c, trange=None, neighbors=0):
        tr = self.get_trace(c, trange=trange, neighbors=neighbors)
        psd = 20 * np.log10(np.abs(np.fft.rfft(tr)) - np.finfo(float).eps)
        return np.fft.rfftfreq(tr.size, self.si), psd

    def get_trace(self, c, trange=None, neighbors=0):
        """
        Get trace according to index, taking into account the orientation of the model
        :param c: trace index
        :param trange: time-range (secs)
        :return: np.array of size (ns, nc)
        """
        trsel = np.arange(-neighbors, neighbors + 1) + int(np.floor(c))
        trsel = trsel[np.logical_and(trsel < self.ntr, trsel >= 0)]
        if trange is not None:
            first_s = int((trange[0] - self.t0) / self.si)
            last_s = int((trange[1] - self.t0) / self.si)
            sl = slice(first_s, last_s)
        else:
            sl = slice(None)
        if self.caxis == 0:
            return np.squeeze(self.data[trsel, sl].T)
        else:
            return np.squeeze(self.data[sl, trsel])

    def set_data(self, data, header=None, si=None, t0=0, x0=0, taxis=1):
        assert (header is not None) or si
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

    @property
    def tscale(self):
        return np.arange(self.ns) * self.si + self.t0


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
    # app = easyqc.qt.create_app()
    eqc = EasyQC._get_or_create(title=title)
    if w is not None:
        eqc.ctrl.update_data(w, h=h, si=si, t0=t0, x0=x0, taxis=taxis)
    eqc.show()
    return eqc


if __name__ == '__main__':
    eqc = viewseis(None)
    app = pg.Qt.mkQApp()
    sys.exit(app.exec_())
