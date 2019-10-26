import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77]

OKVIS = [0.2, 0.205, 0.085, 0.06, 0.05, 0.065, 0.095, 0.11, 0.145, 0.125, 0.120, 0.18, 0.145, 0.196, 0.285, 0.33]
print("RSME of Proposed  : ", np.mean(OKVIS))
se1 = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]

VINS_MONO = [0.26, 0.24, 0.14, 0.09, 0.08, 0.06, 0.05, 0.06, 0.05, 0.055, 0.058, 0.11, 0.101, 0.32, 0.38, 0.54]
print("RSME of VINS_MONO : ", np.mean(VINS_MONO))
se2 = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]

VINS_LOOP = [0.18, 0.17, 0.07, 0.09, 0.11, 0.085, 0.105, 0.096, 0.13, 0.215, 0.22, 0.14, 0.17, 0.31, 0.175, 0.165]
print("RSME of VINS_LOOP : ", np.mean(VINS_LOOP))
se3 = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]

Proposed = [0.11, 0.095, 0.071, 0.075, 0.083, 0.058, 0.066, 0.071, 0.101, 0.091, 0.099, 0.089, 0.091, 0.14, 0.132, 0.17]
print("RSME of Proposed : ", np.mean(Proposed))
se4 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis([0, 80, 0, 0.6])

linestyle = {"linestyle":"--", "linewidth":1, "markeredgewidth":2, "elinewidth":1, "capsize":5}

ax.errorbar(x, OKVIS, yerr = se1,  fmt='-o', mfc='r',  **linestyle, label='OKVIS')
ax.errorbar(x, VINS_MONO, yerr = se2,  fmt='-o', mfc='g',  **linestyle, label='VINS_MONO')
ax.errorbar(x, VINS_LOOP, yerr = se3,  fmt='-^', mfc='b',  **linestyle, label='VINS_LOOP')
ax.errorbar(x, Proposed, yerr = se3,  fmt='-o', mfc='k',  **linestyle, label='Proposed')


#ax.legend(loc='upper center', ncol=2, fancybox=True, shadow=True)

ax.set(xlabel='Distance Travelled (m)', ylabel='RSME (m)')
ax.legend(loc='best', frameon=True)

figure_title = "MH_01_easy"
plt.text(0.5, 1.05, figure_title, horizontalalignment='center', fontsize=18, transform = ax.transAxes)

plt.show()
