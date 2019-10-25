import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import cufflinks as cf


sample_data=pd.read_csv("GT_M01easy.csv")
print(sample_data.head())
print(sample_data.columns)

sample_data = sample_data.rename(columns={'field.point.x': 'XX'})
sample_data = sample_data.rename(columns={'field.point.y': 'YY'})
sample_data = sample_data.rename(columns={'field.point.z': 'ZZ'})

plt.plot(sample_data.XX, sample_data.YY, "g--", linewidth=2, markersize=1)
#plt.plot(sample_data.XX, sample_data.ZZ, "g--", linewidth=2, markersize=1)

#sample_data.plot.scatter(x='XX',y='YY')

plt.legend("easy_MH01")
plt.xlabel("X(meter)")
plt.ylabel("Y(meter)")
plt.title("easy_MH01")
plt.show()

sns.distplot(sample_data['XX'])
sns.distplot(sample_data['YY'])
sns.distplot(sample_data['ZZ'])
plt.show()

sns.heatmap(sample_data.corr(), annot=True)
plt.show()

sns.jointplot(x='XX',y='YY',data=sample_data,kind='scatter')
plt.show()

sns.jointplot(x='XX',y='YY',data=sample_data,kind='hex')
plt.show()

sns.jointplot(x='XX',y='ZZ',data=sample_data,kind='scatter')
plt.show()

sns.jointplot(x='XX',y='ZZ',data=sample_data,kind='reg')
plt.show()

sns.jointplot(x='XX',y='YY',data=sample_data,kind='reg')
plt.show()

sns.lmplot(x='XX',y='YY',data=sample_data)
plt.show()

print(sample_data.isnull())
sns.heatmap(sample_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
#sns.countplot(x='XX', data=sample_data)
#plt.show()

#sns.boxplot(x='XX',y='YY',data=sample_data,palette='winter')
#plt.show()



