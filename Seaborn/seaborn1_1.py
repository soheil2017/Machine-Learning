import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')
print(tips.head())

sns.distplot(tips['total_bill'])
plt.show()

sns.distplot(tips['total_bill'],kde=False,bins=30)
plt.show()

sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')
plt.show()

sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')
plt.show()

sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')
plt.show()

sns.jointplot(x='total_bill',y='tip',data=tips,kind='kde')
plt.show()

sns.pairplot(tips)
plt.show()

sns.pairplot(tips, hue='sex')
plt.show()

sns.pairplot(tips,hue='sex',palette='coolwarm')
plt.show()

sns.boxplot(x="day", y="total_bill", data=tips,palette='rainbow')
plt.show()

sns.boxplot(x="day", y="total_bill", hue="smoker",data=tips, palette="coolwarm")
plt.show()

sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',palette='Set1')
plt.show()

g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.scatter, "total_bill", "tip")
plt.show()

g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g = g.plot(sns.regplot, sns.distplot)
plt.show()
