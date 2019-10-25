import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.keys())
print("="*40)

print(cancer['feature_names'])
print("="*40)

df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df_feat.info())
print("="*40)

print(cancer['target'])
print("="*40)

df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])
print(df_target.head(50))

#Train Test Split
from sklearn.model_selection import train_test_split
X = df_feat
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#Train the Support Vector Classifier
from sklearn.svm import SVC
model = SVC(gamma='auto')
model.fit(X_train,y_train)

#Predictions and Evaluations
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Gridsearch
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)

print(grid.best_params_)
print("="*40)

print(grid.best_estimator_)
print("="*40)

#Then you can re-run predictions on this grid object just like you would with a normal model
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print("="*40)

print(classification_report(y_test,grid_predictions))
print("="*40)
