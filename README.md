# EasyQC
Seismic Viewer Prototype.

## Install Instructions

The easiest way is to install a conda environment and run from sources in development mode.
```
git clone https://github.com/oliche/easyqc.git
conda env create -f ./seisview/env_seisview.yaml
conda activate iblenv
conda develop ./easyqc
```

### Upgrade environment instructions
```
conda env update --file conda_easyqc.yaml --prune
```

Or for a complete clean-up:
```
conda env list
conda env remove -n easyqc
```
And follow the install instructions above.


## Usage Instructions
The goal is to provide an interactive seismic viewer at the python prompt.
NB: if you use `ipython` use the `%gui qt` magic command before !

### Shortcuts
ctrl + A: increase display gain by +3dB
ctrl + Z: deacrease display gain by +3dB
ctrl + P: propagates display accross all windows (same window size, same axis, same gain)

### Minimum working example to display a numpy array.

```python
import numpy as np
import scipy.signal

from easyqc import viewdata

ntr, ns, sr, dx, v1 = (500, 2000, 0.002, 5, 2000)
data = np.zeros((ntr, ns), np.float32)
data[:, 500:600] = scipy.signal.ricker(100, 4)

# create a record with 400 traces and 2500 samples
noise = np.random.randn(ntr, ns) / 10
# create an arbitrary layout of 2 receiver lines of 200 sensors
a, b = np.meshgrid(np.arange(ntr / 2) * 8 + 2000, np.arange(2) * 50 + 5000)
# the header is a dictionary of numpy arrays, each entry being the same length as the number of traces
header = {'receiver_line': b.flatten(), 'receiver_number': a.flatten()}

# show the array with the header
fig0 = viewdata(data, si=.002, h=header, title='clean')
fig1 = viewdata(data + noise, si=.002, h=header, title='noisy')
```
