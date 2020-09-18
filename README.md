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
```
import numpy as np
from easyqc import viewdata

w = np.random.random((2500, 400))
fig = viewdata(w, si=.002)
```
