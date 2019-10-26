import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.DataFrame({'DataSet':['10', '10','20','20', '30', '30', '40', '40', '50', '50'],\
                  'Proposed':[0.1, 4, 0, 4.8, 0, 6, 0, 7.3, 0, 8],
                  'VI-ORB SLAM':[0, 3, 0, 5.1, 0, 6.2, 0, 7.6, 0, 8.1]})

df = df[['DataSet','Proposed','VI-ORB SLAM']]

print(df)

dd=pd.melt(df,id_vars=['DataSet'],value_vars=['Proposed','VI-ORB SLAM'],var_name='Algorithm')
fig = sns.boxplot(x='DataSet',y='value',data=dd,hue='Algorithm', width=0.4)

fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
          ncol=2, fancybox=True, shadow=True)
fig.set(xlabel='Distance Travelled (m)', ylabel='Error (m)')

figure_title = "Laptop"
plt.text(0.5, 1.22, figure_title,
         horizontalalignment='center',
         fontsize=18,
         transform = fig.transAxes)


#plt.grid(True)
plt.show()
