import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sample_data=pd.read_csv('ODROID_vs_laptop.csv')
print(sample_data)

sns.boxenplot(x='DataSet', y='CPU usage', hue='System', data=sample_data)
plt.show()


