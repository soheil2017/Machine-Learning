import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')
print(tips.head())

sns.lmplot(x='total_bill',y='tip',data=tips)
plt.show()

sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex')
plt.show()

sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='coolwarm')
plt.show()

# http://matplotlib.org/api/markers_api.html
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='coolwarm',
           markers=['o','v'],scatter_kws={'s':100})
plt.show()

sns.lmplot(x='total_bill',y='tip',data=tips,col='sex')
plt.show()

sns.lmplot(x="total_bill", y="tip", row="sex", col="time",data=tips)
plt.show()

sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',palette='coolwarm',
          aspect=0.6,height=8)
plt.show()

