import pickle
import numpy as np
import matplotlib.pyplot as plt

rt_info = pickle.load(open('/data2/mengtial/Exp/ArgoVerse/output/rt_ssd512/s1_val/time_all.pkl', 'rb'))
rt_samples = 1e3*np.array(rt_info[0])

plt.hist(rt_samples)
plt.savefig('/data2/mengtial/Exp/ArgoVerse/fig/hist.png')

