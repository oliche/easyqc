# EasyQC
Seismic Viewer for numpy arrays using pyqtgraph.

## Usage Instructions
The goal is to provide an interactive seismic viewer at the python prompt.
NB: if you use `ipython` use the `%gui qt` magic command before !

### Keyboard Shortcuts
-   **ctrl + A**: increase display gain by +3dB 
-   **ctrl + Z**: deacrease display gain by +3dB
-   **ctrl + P**: take screenshot to clipboard
-   **ctrl + P**: propagates display accross all windows (same window size, same axis, same gain)
-   **ctrl + S**: captures screenshot of the plot area in the clipboard
-   **up/down/right/left arrows**: pan using keyboard

### Minimum working example to display a numpy array.

```python
import numpy as np
import scipy.signal

from easyqc.gui import viewseis

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
fig0 = viewseis(data, si=.002, h=header, title='clean')
fig1 = viewseis(data + noise, si=.002, h=header, title='noisy')

```

## Install Instructions

### 1) From pypi using pip:
`pip install easyqc`

### 2) From sources using pip:
I suggest to use a virtual environment and install in development mode (in-place)
```
git clone https://github.com/oliche/easyqc.git
cd easyqc
pip install -e .
```

### 3) From sources using anaconda

I suggest to install a conda environment and run from sources in development mode.
#### Installation
```
git clone https://github.com/oliche/easyqc.git
conda env create -f conda_easyqc.yaml
conda activate iblenv
conda develop ./easyqc
```

#### Update
```
conda env update --file conda_easyqc.yaml --prune
```

Or for a complete clean-up:
```
conda env list
conda env remove -n easyqc
```
And follow the install instructions above.



## Contribution

Pypi Release checklist:
```shell
flake8
rm -fR dist
rm -fR build
python setup.py sdist bdist_wheel
twine upload dist/*
#twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Test wheel:
```shell
virtualenv easyqc --python=3.8
source ./easyqc/bin/activate
pip install easyqc
#pip install -i https://test.pypi.org/simple/ easyqc  # doesnt' seem to install deps
```
