# Display using pyqtgraph
from PyQt5.QtGui import QTransform
import pyqtgraph as pg
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

import numpy as np


class ImShowItem(object):
    def __init__(self, *args, **kwargs):
        self.plotwidget = pg.PlotWidget()
        self.plotitem = self.plotwidget.getPlotItem()
        self.imageitem = pg.ImageItem(np.zeros((2, 2)))
        self.plotwidget.setBackground(pg.mkColor('#ffffff'))
        self.plotwidget.addItem(self.imageitem)
        self.imageitem.getViewBox().setAspectLocked(False)
        self.plotwidget.show()
        self.set_image(*args, **kwargs)
        self.plotwidget.imageshowitem = self

    def set_image(self, image: np.array = None, hscale: np.array = None, vscale: np.array = None,
                  colormap: str = 'viridis'):
        if image is None:
            image = np.zeros((2, 2))
        if hscale is None:
            hscale = [0, 1]
        if vscale is None:
            vscale = [0, 1]
        transform = [hscale[1] - hscale[0], 0., 0., 0., vscale[1] - vscale[0], 0., hscale[0], vscale[0], 1.]
        self.imageitem.setImage(image.T)
        # plotitem.invertY()
        self.imageitem.setTransform(QTransform(*transform))
        self.plotitem.setLimits(xMin=hscale[0], xMax=hscale[-1], yMin=vscale[0], yMax=vscale[-1])
        self.set_colormap(colormap)

    def set_colormap(self, colormap):
        if colormap not in Gradients.keys():
            ValueError(f"{colormap} is not a valid colormap, options are {Gradients.keys()}")
        pgColormap = pg.ColorMap(*zip(*Gradients["viridis"]["ticks"]))
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
