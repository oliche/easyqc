# Display using pyqtgraph
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, QtGui, uic
import pyqtgraph as pg
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

import numpy as np
import scipy.signal


class ImShowSpectrogram(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(ImShowSpectrogram, self).__init__(*args, **kwargs)
        uic.loadUi(Path(__file__).parent.joinpath('spectrogram.ui'), self)
        self.settings = QtCore.QSettings('spectrogram')
        self.imageItem = pg.ImageItem()
        self.plotItem_image.addItem(self.imageItem)
        self._settings2ui()
        self.uiLineEdit_nfft.editingFinished.connect(self._update)
        self.uiLineEdit_nperseg.editingFinished.connect(self._update)
        self.uiLineEdit_noverlap.editingFinished.connect(self._update)
        self.show()
        self.vmin = 0
        self.vmax = None

    @property
    def nfft(self):
        return self.settings.value("nfft", 512)

    @property
    def nperseg(self):
        return self.settings.value("nperseg", 256)

    @property
    def noverlap(self):
        return self.settings.value("noverlap", 128)

    def _settings2ui(self):
        self.uiLineEdit_nfft.setText(str(self.nfft))
        self.uiLineEdit_nperseg.setText(str(self.nperseg))
        self.uiLineEdit_noverlap.setText(str(self.noverlap))

    def _ui2settings(self):
        self.settings.setValue("nfft", int(self.uiLineEdit_nfft.text()))
        self.settings.setValue("nperseg", int(self.uiLineEdit_nperseg.text()))
        self.settings.setValue("noverlap", int(self.uiLineEdit_noverlap.text()))

    def set_data(self, data, fs, colormap='magma'):
        self.data = data
        self.fs = fs
        self.colormap = colormap
        self._update()

    def set_colormap(self, colormap):
        if colormap not in Gradients.keys():
            ValueError(f"{colormap} is not a valid colormap, options are {Gradients.keys()}")
        pgColormap = pg.ColorMap(*zip(*Gradients["magma"]["ticks"]))
        self.imageItem.setLookupTable(pgColormap.getLookupTable())
        if self.vmax is None:
            self.vmax = self.imageItem.getLevels()[1]
        self.imageItem.setLevels((self.vmin, self.vmax))

    def _update(self):
        self._ui2settings()
        fscale, tscale, tf = scipy.signal.spectrogram(
            self.data, fs=self.fs, nperseg=self.nperseg, nfft=self.nfft, window='cosine', noverlap=self.noverlap)
        transform = [tscale[1] - tscale[0], 0., 0., 0., fscale[1] - fscale[0], 0., tscale[0], fscale[0], 1.]
        self.imageItem.setImage(20 * np.log10(tf.T))
        self.imageItem.setTransform(QtGui.QTransform(*transform))
        self.plotItem_image.setLimits(xMin=tscale[0], xMax=tscale[-1], yMin=fscale[0], yMax=fscale[-1])
        self.set_colormap(self.colormap)

    def keyPressEvent(self, e):
        """
        page-up / ctrl + a :  gain up
        page-down / ctrl + z : gain down
        :param e:
        """
        k, m = (e.key(), e.modifiers())
        # page up / ctrl + a
        if k == QtCore.Qt.Key_PageUp or (m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_A):
            self.vmax = self.vmax - 3
            self.imageItem.setLevels((self.vmin, self.vmax))
        # page down / ctrl + z
        elif k == QtCore.Qt.Key_PageDown or (m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_Z):
            self.vmax = self.vmax + 3
            self.imageItem.setLevels((self.vmin, self.vmax))
        print(self.vmax)


class ImShowItem(object):
    """
    This class is a wrapper around the pyqtgraph ImageItem class to display images in a pyqtgraph plotwidget.
        :param image:
        :param hscale:
        :param vscale:
        :param colormap:
        :return:
    """
    def __init__(self, *args, **kwargs):
        self.plotwidget = pg.PlotWidget()
        self.plotitem = self.plotwidget.getPlotItem()
        self.imageitem = pg.ImageItem(np.zeros((2, 2)))
        self.plotwidget.setBackground(pg.mkColor('#ffffff'))
        self.plotwidget.addItem(self.imageitem)
        self.imageitem.getViewBox().setAspectLocked(False)
        self.plotwidget.show()
        self.plotwidget.imageshowitem = self
        self.__post_init__(*args, **kwargs)

    def __post_init__(self, *args, **kwargs):
        """
        This can be overloaded by child classes
        """
        self.set_image(*args, **kwargs)

    def set_image(self, image: np.array = None, hscale: np.array = None, vscale: np.array = None,
                  colormap: str = 'magma'):
        """
        :param image:
        :param hscale:
        :param vscale:
        :param colormap:
        :return:
        """
        if image is None:
            image = np.zeros((2, 2))
        if hscale is None:
            hscale = [0, 1]
        if vscale is None:
            vscale = [0, 1]
        transform = [hscale[1] - hscale[0], 0., 0., 0., vscale[1] - vscale[0], 0., hscale[0], vscale[0], 1.]
        self.imageitem.setImage(image.T)
        # plotitem.invertY()
        self.imageitem.setTransform(QtGui.QTransform(*transform))
        self.plotitem.setLimits(xMin=hscale[0], xMax=hscale[-1], yMin=vscale[0], yMax=vscale[-1])
        self.set_colormap(colormap)

    def set_colormap(self, colormap):
        if colormap not in Gradients.keys():
            ValueError(f"{colormap} is not a valid colormap, options are {Gradients.keys()}")
        pgColormap = pg.ColorMap(*zip(*Gradients["magma"]["ticks"]))
        self.imageitem.setLookupTable(pgColormap.getLookupTable())


def imshow(image: np.array, hscale: np.array = None, vscale: np.array = None,
           colormap: str = 'viridis', imshowitem: ImShowItem = None) -> ImShowItem:
    """
    :param image: axis 0 is the vertical direction, axis 1 is the horizontal direction
    :param hscale:
    :param vscale:
    :param imshowitem:
    :param colormap:
    :return:
    """
    if imshowitem is None:
        imshowitem = ImShowItem(image, hscale, vscale, colormap)
    else:
        imshowitem.set_image(image, hscale, vscale, colormap)
    return imshowitem
