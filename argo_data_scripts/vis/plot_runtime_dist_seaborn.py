import pickle
from os.path import join, basename

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


palettes = ['#293E5C', '#FAD893', '#FF550D', '#13C4E8', '#FA72B4', '#70D4BC', '#5836C9', '#42b7cf']

runtime = r'D:\Data\Exp\ArgoVerse1.1-c3-eta0\runtime-zoo\1080ti\retina50_s0.5.pkl'

rt_samples = pickle.load(open(runtime, 'rb'))['samples']
rt_samples = 1e3*np.asarray(rt_samples)

plt.rcParams.update({'font.size': 19})

sns.set_style('darkgrid')
sns.distplot(rt_samples, color=palettes[-1])

plt.xlim(45, 80)
plt.ylim(0, 0.2)
plt.xlabel('Runtime (ms)')
plt.ylabel('Density')

fig = plt.gcf()
fig.set_size_inches(9, 5)


plt.savefig("runtime-hist.pdf", bbox_inches='tight')
plt.show()
