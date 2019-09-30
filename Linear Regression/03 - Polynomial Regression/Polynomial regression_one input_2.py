# Linear Regression With statsmodelsAdvanced
#ones have been considered
# 𝑓(𝑥) = 𝑏₀ + 𝑏₁x + 𝑏₂x²     Polynomial for one input
#Quadratic regression, or regression with a second order polynomial

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([[5], [15], [25], [35], [45], [55], [62]])
y = np.array([[15], [11], [2], [8], [25], [32], [41]])

quadratic_featurizer = PolynomialFeatures(degree=2)
x_ = quadratic_featurizer.fit_transform(x)
print("modified x =", "\n", x_, "\n")

x0 = x_[:,[0]]
x1 = x_[:,[1]]
x2 = x_[:,[2]]      # x2=𝑥₁²

#------------------------------------------------------------------------
model = LinearRegression().fit(x, y)
xx = np.linspace(0, 90, 100)
xx = xx.reshape((-1, 1))
#https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
yy = model.predict(xx.reshape(xx.shape[0], 1))  #xx.shape[0] --> row numbers of xx
                                                #xx.shape[1]--> column numbers of xx
#------------------------------------------------------------------------
model_quadratic = LinearRegression().fit(x_, y)   #plonominal equation
xx_quadratic = quadratic_featurizer.transform(xx)
print(xx_quadratic)
#------------------------------------------------------------------------

#------------------------------------------------------------------------
plt.plot(xx, yy)
plt.plot(xx, model_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 60, 0, 40])
plt.scatter(x, y)
plt.grid(True)
plt.show()

y_pre = model.predict(x)
print("predicted y = y_pre", y_pre, sep="\n   ")

print('Simple linear regression R^2 :', model.score(x, y), "\n")
# R-squared measures how well the observed values of the response variables are
# predicted by the model. More concretely, r-squared is the proportion of the
# variance in the response variable that is explained by the model.
#print(model.intercept_(x, y))
#print(model.coef_(x, y))
print('Quadratic regression R^2 :', model_quadratic.score(x_, y), "\n")
print("b0=", model_quadratic.intercept_, "\n", "b =", model_quadratic.coef_, "\n")