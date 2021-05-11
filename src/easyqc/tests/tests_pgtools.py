import numpy as np
from scipy.signal import spectrogram, chirp
import matplotlib.pyplot as plt  # noqa

from easyqc import pgtools


def test_pgtools_imshow():
    # try to instantiate an ImShowItem
    pgtools.ImShowItem()

    # Create spectrogram
    t = np.arange(2000) * .002
    w = chirp(t, f0=4, f1=80, t1=3)
    fscale, tscale, tf = spectrogram(w, fs=500, nperseg=256, nfft=512, window='cosine', noverlap=250)

    # Display using matplotlib for reference
    # import matplotlib.pyplot as plt
    # plt.imshow(tf, extent=[tscale[0], tscale[-1], fscale[0], fscale[-1]], aspect='auto', origin='lower')
    imshowitem = pgtools.imshow(tf, tscale, fscale)

    w = chirp(t, f0=4, f1=160, t1=3)
    fscale, tscale, tf = spectrogram(w, fs=500, nperseg=256, nfft=512, window='cosine', noverlap=250)
    imshowitem.set_image(tf, tscale, fscale)
