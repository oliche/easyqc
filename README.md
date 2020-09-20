# seisview
Seismic Viewer Prototype.

## Install Instructions

The easiest way is to install a conda environment and run from sources in development mode.
```
git clone https://github.com/oliche/seisview.git
conda env create -f ./seisview/env_seisview.yaml
conda activate iblenv
conda develop ./seisview
```

### Upgrade environment instructions
```
conda env update --file env_seisview.yaml --prune
```

Or for a complete clean-up:
```
conda env list
conda env remove -n iblenv
```
And follow the install instructions above.


## Usage Instructions
The goal is to provide an interactive seismic viewer at the python prompt.
NB: if you use `ipython` use the `%gui qt` magic command before !

here is a minimally working example to display a numpy array.
```python
import numpy as np
from easyqc import viewdata

# create a record with 400 traces and 2500 samples
w = np.random.random((400, 2500)) - 0.5

# create an arbitrary layout of 2 receiver lines of 200 sensors
a, b = np.meshgrid(np.arange(200) * 8 + 2000, np.arange(2) * 50 + 5000)
# the header is a dictionary of numpy arrays, each entry being the same length as the number of traces
header = {'receiver_line': b.flatten(), 'receiver_number': a.flatten()}

# show the array with the header
fig = viewdata(w, si=.002, h=header)
```
