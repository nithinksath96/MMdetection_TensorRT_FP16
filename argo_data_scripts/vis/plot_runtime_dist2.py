import pickle
from os.path import join, basename

import numpy as np
import matplotlib.pyplot as plt


palettes = ['#293E5C', '#FAD893', '#FF550D', '#13C4E8', '#FA72B4', '#70D4BC', '#5836C9']

runtime =[
    r'D:\Data\Exp\ArgoVerse1.1-c3-eta0\runtime-zoo\1080ti\retina50_s0.5.pkl',
    r'D:\Data\Exp\ArgoVerse1.1-c3-eta0\runtime-zoo\1080ti\mrcnn50_nm_s0.5.pkl',
]

names = [
    'Single Stage (RetinaNet)',
    'Two Stage (Mask R-CNN)',
]




n = len(names)
fig, axs = plt.subplots(1, n)
axs = axs.flatten()

for i in range(n):
    ax = axs[i]
    rt_samples = pickle.load(open(runtime[i], 'rb'))['samples']
    rt_samples = 1e3*np.asarray(rt_samples)
    ax.hist(rt_samples)
    ax.set_title('(' + chr(ord('a') + i) + ') ' + names[i])
    ax.set_xlabel('Runtime (ms)')
    ax.set_ylabel('Count')


fig.set_size_inches(10, 6)
plt.rcParams.update({'font.size': 22})
# plt.subplots_adjust(wspace=0.7, hspace=0.5, top=0.95, bottom=0.09, left=0.045, right=0.95)

plt.savefig("runtime-hist.pdf", bbox_inches='tight')
plt.show()
