# we will split the data into training and testing sets,
# train the regressor,  and evaluate its predictions:
# By default, 25 percent of the data is assigned to the test set

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import cross_val_score


df = pd.read_csv('winequality-red.csv', sep=';')
print(df.columns, "\n")

X = df[list(df.columns)[:-1]]
print(X)
y = df['quality']
print(y)

regressor = LinearRegression()
scores = cross_val_score(regressor, X, y, cv=5)

print("  mean of the scores is",scores.mean(), " and r-squared scores range", scores )
#The mean of the scores, 0.29, is a better estimation of the estimator's predictive
# power than the r-squared score produced from a single train / test split.
print("------------------------------------------------")
