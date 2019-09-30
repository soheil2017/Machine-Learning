 # https://realpython.com/linear-regression-in-python/#regression
# https://zerotohero.ir/article/python/%D8%B1%D8%B3%D9%85-%D9%86%D9%85%D9%88%D8%AF%D8%A7%D8%B1-%D9%88-%D9%85%D8%B5%D9%88%D8%B1-%D8%B3%D8%A7%D8%B2%DB%8C-%D8%AF%D8%A7%D8%AF%D9%87%E2%80%8C%D9%87%D8%A7-%D8%AF%D8%B1-%D9%BE%D8%A7%DB%8C%D8%AA/
#  one input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([6, 13, 20, 30, 40, 48]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38]).reshape((-1, 1))
print(x, "\n")
print(y)

new_model = LinearRegression().fit(x,y)

y_pre = new_model.predict(x)
print("predicted y = y_pre", y_pre, sep="\n")
print("b0=", new_model.intercept_, "\n", "b1=", new_model.coef_)   # new_model.intercept_= b0, new_model.coef_=b2

#plt.plot(x, y, "ro", x, 5.633+0.54*x, "b-")
#plt.xlabel("Input X")
#plt.ylabel("output Y")
#plt.title("linear regression")
#plt.show()

r_sq = new_model.score(x, y)            # R^2= coefficient of determination
print('R^2:', r_sq)

plt.plot(x, y, "r+", x, new_model.predict(x), "b-", x, y_pre, "g+")
plt.legend(["input", "function", "predicted"])

#plt.legend(loc="lower right")

#plt.plot(x, y, "r+", label="inputs")
#plt.plot(x, 5.617+0.619*x, "b-", label="line")
#plt.plot(x, y_pre, "g+", label="estimated")

plt.xlabel("Input X")
plt.ylabel("output Y")
plt.title("linear regression")

print("new data prediction:", 5.617+0.619*35)


plt.show()

x_new = np.arange(5).reshape((-1, 1))
print(x_new, "\n")
y_new = new_model.predict(x_new)
print("y new is:","\n", y_new)
#y_new = new_model.predict(x_new)
#print("new prediction is:", y_new)