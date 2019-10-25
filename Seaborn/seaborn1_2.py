import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')
print(tips.head())
sns.boxenplot(x='day', y='total_bill', hue='smoker', data=tips)
plt.show()

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
print(tips.head())
print(flights.head())

sns.boxenplot(x='day', y='total_bill', hue='smoker', data=tips)
plt.show()

tc = tips.corr()
print(tc, "\n")
sns.heatmap(tc, annot=True, cmap='coolwarm')
plt.show()

fp = flights.pivot_table(index='month', columns='year', values='passengers')
print(fp)

sns.heatmap(fp, cmap='magma', linecolor='white', linewidth=3)
plt.show()

sns.clustermap(fp, cmap='coolwarm', standard_scale=1)
plt.show()

