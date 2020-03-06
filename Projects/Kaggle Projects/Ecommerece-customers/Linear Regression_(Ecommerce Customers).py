import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customer = pd.read_csv('Ecommerce Customers.csv')
print(customer.head())
print("="*40)

print(customer.info())
print("="*40)

print(customer.describe())
print("="*40)

print(customer.columns)
print("="*40)

print(customer.corr())
print("="*40)

#**Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(data=customer,x='Time on Website',y='Yearly Amount Spent')
plt.show()

#** Do the same but with the Time on App column instead. **
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customer)
plt.show()

#** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customer)
plt.show()

#Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.
sns.pairplot(customer)
plt.show()
#Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?
## Length of Membership

#*Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customer)
plt.show()

#Training and Testing Data using Scikit learn
print(customer.columns)

y = customer['Yearly Amount Spent']
X = customer[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
#** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


##########   fitting the model   ##############
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

# #Model Evaluation
print(lm.intercept_)
print("="*40)
print(lm.coef_)
print("="*40)

# #Predictions from our Model
#** Use lm.predict() to predict off the X_test set of the data.**
predictions = lm.predict(X_test)
print(predictions)
print("="*40)

#** Create a scatterplot of the real test values versus the predicted values. **
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

#Residuals
sns.distplot((y_test-predictions))
plt.show()

# #Regression Evaluation Metrics
# ## Comparing these metrics:
# #Mean Absolute Error (MAE) is the easiest to understand, because it's the average error.
# #Mean Squared Error (MSE) is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# #Root Mean Squared Error (RMSE)  is even more popular than MSE, because RMSE is interpretable in the "y" units.
# #All of these are loss functions, because we want to minimize them.

from sklearn import metrics
print("Mean Absolute Error (MAE) is: {}".format(metrics.mean_absolute_error(y_test,predictions)))
print("="*40)

print("Mean Squared Error (MSE) is: {}".format(metrics.mean_squared_error(y_test,predictions)))
print("="*40)

print("Root Mean Squared Error (RMSE) is: {}".format(np.sqrt(metrics.mean_squared_error(y_test,predictions))))
print("="*40)

#We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development?
# Or maybe that doesn't even really matter, and Membership Time is what is really important.
# Let's see if we can interpret the coefficients at all to get an idea.
#** Recreate the dataframe below. **

coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)
print("="*40)

#** How can you interpret these coefficients? **

##Interpreting the coefficients:
#Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
#Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
#Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
#Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.

#Do you think the company should focus more on their mobile app or on their website?

#This is tricky, there are two ways to think about this:
# Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better.
# This sort of answer really depends on the other factors going on at the company,
# you would probably want to explore the relationship between Length of Membership and the App or the Website before coming to a conclusion!