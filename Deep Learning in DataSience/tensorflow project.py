import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('bank_note_data.csv')
print(data.head())

sns.countplot(x='Class',data=data)
plt.show()
sns.pairplot(data,hue='Class')
plt.show()

#Data Preparation
#When using Neural Network and Deep Learning based systems, it is usually a good idea to Standardize your data,
# this step isn't actually necessary for our particular data set, but let's run through it for practice!

#Standard Scaling
from sklearn.preprocessing import StandardScaler
#Create a StandardScaler() object called scaler.
scaler = StandardScaler()
#Fit scaler to the features.
scaler.fit(data.drop('Class',axis=1))

#Use the .transform() method to transform the features to a scaled version.
scaled_features = scaler.fit_transform(data.drop('Class',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
print(df_feat.head())

#Train Test Split
#Create two objects X and y which are the scaled feature values and labels respectively.
X = df_feat
y = data['Class']
#** Use SciKit Learn to create training and testing sets of the data as we've done in previous lectures:**
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Tensorflow
import tensorflow as tf
#** Create a list of feature column objects using tf.feature.numeric_column() as we did in the lecture**
print(df_feat.columns)

image_var = tf.feature_column.numeric_column("Image.Var")
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy =tf.feature_column.numeric_column('Entropy')

feat_cols = [image_var,image_skew,image_curt,entropy]
#** Create an object called classifier which is a DNNClassifier from learn. Set it to have 2 classes and a [10,20,10] hidden unit layer structure:**
classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)

#** Now create a tf.estimator.pandas_input_fn that takes in your X_train, y_train, batch_size and set shuffle=True.
# You can play around with the batch_size parameter if you want, but let's start by setting it to 20 since our data isn't very big. **
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)

#** Now train classifier to the input function. Use steps=500. You can play around with these values if you want!**
classifier.train(input_fn=input_func,steps=500)

#Model Evaluation
#** Create another pandas_input_fn that takes in the X_test data for x.
# Remember this one won't need any y_test info since we will be using this for the network to create its own predictions.
# Set shuffle=False since we don't need to shuffle for predictions.**
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
#** Use the predict method from the classifier model to create predictions from X_test **
note_predictions = list(classifier.predict(input_fn=pred_fn))
print(note_predictions[0])

final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])

#** Now create a classification report and a Confusion Matrix. Does anything stand out to you?**
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,final_preds))
print(classification_report(y_test,final_preds))

#Optional Comparison
#** You should have noticed extremely accurate results from the DNN model. Let's compare this to a Random Forest Classifier for a reality check!**
#Use SciKit Learn to Create a Random Forest Classifier and compare the confusion matrix and classification report to the DNN model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
print(rfc.fit(X_train,y_train))
rfc_preds = rfc.predict(X_test)
print(classification_report(y_test,rfc_preds))
print(confusion_matrix(y_test,rfc_preds))
