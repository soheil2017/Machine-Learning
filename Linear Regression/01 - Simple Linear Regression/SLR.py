
#SLR  = Simple linear regression

# Importing Library
import numpy as np    #for mathematical calculation 
import matplotlib.pyplot as plt #for ploting nice chat and graph
import pandas as pd  #for managing data set
import seaborn as sns

# Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')

print(dataset.head())
print("="*40)

print(dataset.columns)
print("="*40)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values

# Spliting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Fitting simple linear regression to the tranning set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Training
print("b0=", model.intercept_, "\n", "b1=", model.coef_)   # new_model.intercept_= b0, new_model.coef_=b2

predictions = model.predict(X)
print("the predicted y = ", predictions, sep="\n   ")
print("="*40)

# Predicting the test set result
predictions = model.predict(X_test)
print(predictions)
print("="*40)

plt.scatter(y_test, predictions)
plt.show()

sns.distplot((y_test-predictions))  #histogram of residuals
plt.show()

# Visualising the test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train,model.predict(X_train), color = 'blue')
plt.title('Salary vs Experiance (test set)')
plt.xlabel('year of experiance')
plt.ylabel('salary')
plt.show()

