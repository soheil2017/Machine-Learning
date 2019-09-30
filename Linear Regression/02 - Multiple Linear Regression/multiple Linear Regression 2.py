#https://archive.ics.uci.edu/ml/datasets.php

import numpy as np
import statsmodels.api as sm

x = np.array([[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]])
y = np.array([4, 5, 20, 14, 32, 22, 38, 43]).reshape((-1, 1))
x, y = np.array(x), np.array(y)
print(x, "\n", y)
print("="*40)

#You need to add the column of ones to the inputs if you want statsmodels to calculate the intercept 𝑏₀.
# It doesn’t takes 𝑏₀ into account by default.

x = sm.add_constant(x) # it add one column
print(x)
print("="*40)

# Create a model and fit it
model = sm.OLS(y, x)  #ordinary least squares
results = model.fit()

print(results.summary())
print("="*40)

print('coefficient of determination:', results.rsquared)
print("="*40)

print('adjusted coefficient of determination:', results.rsquared_adj)
#(𝑅² corrected according to the number of input features)
print("="*40)

print('regression coefficients:', results.params) #params refers the array with 𝑏₀, 𝑏₁, and 𝑏₂ respectively.
print("="*40)

print('predicted response:', results.fittedvalues, sep='\n')
print("="*40)

print('predicted response:', results.predict(x), sep='\n')
print("="*40)

# predictions with new regressors
x_new = sm.add_constant(np.arange(10).reshape((-1, 2)))
print(x_new)
print("="*40)

y_new = results.predict(x_new)
print(y_new)
print("="*40)

