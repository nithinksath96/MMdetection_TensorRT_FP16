'''
This script is for testing the map_roi_levels in single_level.py in mmdetection
'''

import numpy as np
import matplotlib.pyplot as plt

nl = 3
fs = 28
s = np.arange(0, 500, 1)

tl = np.floor(np.log2(s/fs + 1e-6))
tl = np.clip(tl, 0, nl-1)

plt.plot(s, tl)
plt.show()