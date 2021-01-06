import csv
from os.path import join, basename

import numpy as np
import matplotlib.pyplot as plt


palettes = ['#293E5C', '#b8e366', '#7b5be3', '#13C4E8', '#FAD893',  '#FF550D', '#FA72B4', '#70D4BC']

data_path = r'D:\Research\Deva\Documents\Streaming\Fig\accuracy-vs-latency.csv'
methods = []
runtime = []
accu = []

with open(data_path) as f:
    rows = csv.DictReader(f)

    for row in rows:
        if row['XY'] == 'Runtime':
            methods.append(row['Method'])
            runtime.append([
                float(row['0.25']),
                float(row['0.5']),
                float(row['0.75']),
                float(row['1']),
            ])
        else:
            accu.append([
                float(row['0.25']),
                float(row['0.5']),
                float(row['0.75']),
                float(row['1']),
            ])


fig = plt.gcf()
fig.set_size_inches(9, 5)
plt.rcParams.update({'font.size': 15.9})

fig.set_size_inches(8, 6)
plt.rcParams.update({'font.size': 17})

n_method = len(methods)
for i in range(n_method):
    plt.plot(runtime[i], accu[i], '-o', label=methods[i], color=palettes[i], linewidth=3, markersize=8)

plt.xlabel('Runtime (ms)')
plt.ylabel('Offline AP')
# plt.axis('square')
plt.legend()

plt.savefig("accu-vs-runtime.pdf", bbox_inches='tight')
plt.show()
