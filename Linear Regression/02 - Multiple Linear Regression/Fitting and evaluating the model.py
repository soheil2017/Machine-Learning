# we will split the data into training and testing sets,
# train the regressor,  and evaluate its predictions:

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('winequality-red.csv', sep=';')
print(df.columns, "\n")
print("="*50)

print(df.describe())
print("="*50)



plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()

##########-----------Fitting and evaluating the model--------#########
X = df[list(df.columns)[:-1]]
print(X)
print("="*40)

y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)
print(predictions)
print("="*40)

print('R-squared R2:', regressor.score(X_test, y_test))
#The r-squared score of 0.38 indicates that 35 percent of the variance in the test set  is explained by the model.
print("="*40)

###########- using cross-validation to produce a better estimate of the estimator's performance
from sklearn.model_selection import cross_val_score

X = df[list(df.columns)[:-1]]
y = df['quality']

regressor = LinearRegression()
scores = cross_val_score(regressor, X, y, cv=5)

print(scores.mean(), scores)
print("="*40)
#The mean of the scores, 0.29, is a better estimate of the estimator's predictive power
# than the r-squared score produced from a single train / test split.
