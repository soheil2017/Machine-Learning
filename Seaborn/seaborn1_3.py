import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
print(iris.head())


g = sns.PairGrid(iris)
g.map(plt.scatter)
plt.show()

g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
plt.show()

sns.pairplot(iris)
plt.show()

sns.pairplot(iris,hue='species',palette='rainbow')
plt.show()


