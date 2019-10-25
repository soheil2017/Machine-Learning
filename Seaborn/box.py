import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')
print(tips.head())
sns.boxenplot(x='day', y='total_bill', hue='smoker', data=tips)
plt.show()