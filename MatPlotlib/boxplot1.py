import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.DataFrame({'DataSet':['A7', 'A7','14.0','14.0', '21.0', '21.0', '28.0', '28.0', '35.0', '35.0'],\
                  'Proposed':[0.1, 4, 0, 4.8, 0, 6, 0, 7.3, 0, 8],
                  'VI-ORB SLAM':[0, 3, 0, 5.1, 0, 6.2, 0, 7.6, 0, 8.1],
                   'EKF-map': [0.12, 3.5, 0, 5.3, 0, 6.1, 0, 7.5, 0, 8],
                   'Shen': [0.15, 3.3, 0, 5, 0, 6.15, 0, 9, 0, 8.3]})
#df = df[['DataSet','Proposed','VI-ORB SLAM', 'EKF-map', 'Shen']]
print(df)

dd = pd.melt(df,id_vars=['DataSet'],value_vars=['Proposed','VI-ORB SLAM','EKF-map', 'Shen'],var_name='Algorithm')
fig = sns.boxplot(x='DataSet',y='value',data=dd,hue='Algorithm')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=4, fancybox=True, shadow=True)
fig.set(xlabel='Distance Travelled (m)', ylabel='Error (m)')

figure_title = "ODROID"
plt.text(0.5, 1.22, figure_title,
         horizontalalignment='center',
         fontsize=18,
         transform = fig.transAxes)


#plt.grid(True)
plt.show()
