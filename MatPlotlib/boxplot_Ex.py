#https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.DataFrame({'DataSet':['MV_01','MV_01','MV_02','MV_02', 'MV_03', 'MV_03', 'MV_04', 'MV_04', 'MV_05', 'MV_05'],\
                  'ODROID':[2, 5, 3, 5, 8, 6, 8, 9, 6, 8],'laptop':[4, 2, 2.5, 3.5, 4, 5, 6, 5, 5.5, 9]})
df = df[['DataSet','ODROID','laptop']]

print(df)

dd=pd.melt(df,id_vars=['DataSet'],value_vars=['ODROID','laptop'],var_name='System')
sns.boxplot(x='DataSet',y='value',data=dd,hue='System')
plt.show()
