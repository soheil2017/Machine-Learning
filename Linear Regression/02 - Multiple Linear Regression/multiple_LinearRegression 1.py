import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7],    [9],    [13],    [17.5],  [18]]

model = LinearRegression()
model.fit(X,y)

X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11],   [8.5],  [15],    [18],    [11]]

predictions = model.predict(X_test)
print(predictions)
print("="*40)

# for i, prediction in enumerate(predictions):
#     print(prediction, y_test[i])

# r-squared is the proportion of the variance in the response variable that is explained by the model.
# An r-squared score of one indicates that the response variable can be predicted without any error using the model.
# An r-squared score of one half indicates that half of the variance in the response variable can be predicted using the model.
r_sq = model.score(X_test, y_test)                       # R^2= coefficient of determination
print('R^2:', r_sq)
print("="*40)
