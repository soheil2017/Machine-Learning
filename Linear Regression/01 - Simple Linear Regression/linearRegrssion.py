import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  #The class sklearn.linear_model.LinearRegression
                                                   # will be used to perform linear and
                                                   # polynomial regression and make predictions accordingly.
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([[5], [20], [14], [32], [22], [38]])

model = LinearRegression().fit(x,y)              #With .fit(), you calculate the optimal values

predictions = model.predict(x)

r_sq = model.score(x, y)                       # R^2= coefficient of determination
print('R^2:', r_sq)
print("="*40)

print("b0=", model.intercept_, "\n", "b1=", model.coef_)
print("="*40)

plt.scatter(x, y)
plt.plot(x, model.predict(x) , "r-")
plt.xlabel("Input X")
plt.ylabel("output Y")
plt.title("linear regression")
plt.show()

