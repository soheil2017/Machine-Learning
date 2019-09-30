# Linear Regression With statsmodelsAdvanced
#ones have been considered
# 𝑓(𝑥) = 𝑏₀ + 𝑏₁x + 𝑏₂x²     Polynomial for one input
#Quadratic regression, or regression with a second order polynomial
# a special case of multiple linear regression that adds terms with degrees greater than one to the model.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_train = [[6], [8], [10], [14],   [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6],  [8],   [11], [16]]
y_test = [[8], [12], [15], [18]]

model = LinearRegression()
model.fit(X_train, y_train)

xx = np.linspace(0, 26, 100)
yy = model.predict(xx.reshape(xx.shape[0], 1))


quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.fit_transform(X_test)

model_quadratic = LinearRegression()
model_quadratic.fit(X_train_quadratic, y_train)

xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx. shape[0], 1))

plt.plot(xx, yy, c='b', linestyle= '--')
plt.plot(xx, model_quadratic.predict(xx_quadratic) ,c='r', linestyle='--')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()

print(X_train)
print("="*50)

print(X_train_quadratic)
print("="*50)

print(X_test)
print("="*50)

print(X_test_quadratic)
print("="*50)

print('Simple linear regression r-squared', model.score(X_test, y_test))
print("="*50)

print('Quadratic regression r-squared', model_quadratic. score(X_test_quadratic, y_test))
print("="*50)
