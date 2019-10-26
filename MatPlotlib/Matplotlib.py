import pandas as pd
import matplotlib.pylab as plt

df = pd.read_csv('winequality-red.csv', sep=';')

print(df.columns)
print(df.describe())
print(df.head())     #first 5 rows

plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.grid(True)
plt.show()

plt.scatter(df['volatile acidity'], df['quality'])
plt.grid(True)
plt.show()
# Ridge regression calculation page 40 book

