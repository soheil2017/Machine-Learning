#https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
"""
=============================
Grouped bar chart with labels
=============================

Bar charts are useful for visualizing counts, or summary statistics
with error bars. This example shows a ways to create a grouped bar chart
with Matplotlib and also how to annotate bars with labels.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


men_means= (0.2, 0.3, 0.31, 0.16, 0.22)
women_means = (0.18, 0.27, 0.21, 0.12, 0.19)

ind = np.arange(len(men_means))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, men_means, width,
                label='Proposed')
rects2 = ax.bar(ind + width/2, women_means, width,
                label='VI ORB-SLAM')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Error (m)')
ax.set_xlabel('Distance Travelled (m)')
#ax.set_xticks(ind)
ax.set_xticklabels(('7.0', '14.0', '21.0', '28.0', '35.0', '42'))
ax.legend()


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


autolabel(rects1, "left")
autolabel(rects2, "right")
#plt.grid(True)
fig.tight_layout()

plt.show()


#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods and classes is shown
# in this example:

matplotlib.axes.Axes.bar
matplotlib.pyplot.bar
matplotlib.axes.Axes.annotate
matplotlib.pyplot.annotate
