import matplotlib.pyplot as plt
from statistics import variance
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([6, 8, 10, 14, 18]).reshape((-1, 1))
y = np.array([7, 9, 13, 17.5, 18]).reshape((-1, 1))

model = LinearRegression().fit(X, y)

print("b0=", model.intercept_, "\n", "b1=", model.coef_)   # new_model.intercept_= b0, new_model.coef_=b2

y_pre = model.predict(X)
print("the predicted y = ", y_pre, sep="\n   ")

print("SS_res:", np.mean((y_pre-y)**2), "\n  ")   #residual sum of squares cost function

#-----------------------------------------------------------------------------------------
#Variance is a measure of how far a set of values is spread out. If all of the numbers in the set are equal, the variance of the set is zero.
# A small variance indicates that the numbers are near the mean of the set, while a set containing numbers that are far from the mean and each other will have a large variance

print("Variance X method 1 is ", np.var(X, ddof=1), "\n")              # variance calculation
                                                             # The ddof keyword parameter can be used to
                                                     # set Bessel's correction to calculate the sample variance:
print("variance x from method 2: ", variance(([6, 8, 10, 14, 18])), "\n")
              # when you import statistics library, you would not need to ddof=1
                   # in method 2 X should be a list or array, not matrix

#Covariance is a measure of how much two variables change together. If the value of the variables increase together, their covariance is positive.
# If one variable tends to increase while the other decreases, their covariance is negative.
# If there is no linear relationship between the two variables, their covariance will be equal to zero
print("Covariance (X, Y) is: ", np.cov([6, 8, 10, 14, 18], [7, 9, 13, 17.5, 18])[0][1])
#print(np.cov(X, y)[0][1])  it works but X and y should be without np.reshape(-1, 1)
#---------------------------------------------------------------------------------------------
#Evaluating the model
X_test = [[8], [9], [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]]
model1 = LinearRegression()
model1.fit(X, y)
r_sq = model1.score(X_test, y_test)  # R^2= coefficient of determination - r-squared
print('R^2:', r_sq)
#An r-squared score of 0.6620 indicates that a large proportion of the variance in the test instances' prices is explained by the model.
#---------------------------------------------------------------------------------------------

plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, 'r*', X, 1.97+0.97*X, "b-")
#plt.plot(t, 1.97+0.97*t )
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()

