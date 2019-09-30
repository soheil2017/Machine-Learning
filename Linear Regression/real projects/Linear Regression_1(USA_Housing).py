import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('USA_Housing.csv')
print(df.head())
print("="*40)

print(df.info())
print("="*40)

print(df.describe())
print("="*40)

print(df.columns)
print("="*40)

sns.pairplot(df)
plt.show()

sns.distplot(df['Price'])  #distrubution of column
plt.show()

print(df.corr())
print("="*40)

sns.heatmap(df.corr(), annot=True)
plt.show()

#training the model using scikit-learn
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
# in X, we should ignore "Price"(because it is a y) and address (it is string)
y = df['Price']

from sklearn.model_selection import train_test_split

#Creating and Training the Model
X_train, X_test, y_train, y_test = train_test_split(
                   X, y, test_size=0.4, random_state=101)

print(X_train)
print("="*40)
print(y_train)
print("="*40)

#fitting the model
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

#Model Evaluation
print(lm.intercept_)
print("="*40)
print(lm.coef_)
print("="*40)

print(X_train.columns)
print(lm.coef_)
print("="*40)

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)
print("="*40)
# Interpreting the coefficients:
# Holding all other features fixed, a 1 unit increase in Avg. Area Income is associated with an *increase of $21.52 *.
# Holding all other features fixed, a 1 unit increase in Avg. Area House Age is associated with an *increase of $164883.28 *.
# Holding all other features fixed, a 1 unit increase in Avg. Area Number of Rooms is associated with an *increase of $122368.67 *.
# Holding all other features fixed, a 1 unit increase in Avg. Area Number of Bedrooms is associated with an *increase of $2233.80 *.
# Holding all other features fixed, a 1 unit increase in Area Population is associated with an *increase of $15.15 *.

# Analyse from other way
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.keys())
#dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
print(boston['target'])  # ['data'] or ['feature_names'] ['filename'] ['DESCR']
print("="*40)

#Predictions from our Model
predictions = lm.predict(X_test)
print(predictions)
print("="*40)

plt.scatter(y_test, predictions)
plt.show()

sns.distplot((y_test-predictions))  #histogram of residuals
plt.show()

#Regression Evaluation Metrics
## Comparing these metrics:
#Mean Absolute Error (MAE) is the easiest to understand, because it's the average error.
#Mean Squared Error (MSE) is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
#Root Mean Squared Error (RMSE)  is even more popular than MSE, because RMSE is interpretable in the "y" units.
#All of these are loss functions, because we want to minimize them.
from sklearn import metrics
print("Mean Absolute Error (MAE) is: {}".format(metrics.mean_absolute_error(y_test,predictions)))
print("="*40)

print("Mean Squared Error (MSE) is: {}".format(metrics.mean_squared_error(y_test,predictions)))
print("="*40)

print("Root Mean Squared Error (RMSE) is: {}".format(np.sqrt(metrics.mean_squared_error(y_test,predictions))))
print("="*40)

