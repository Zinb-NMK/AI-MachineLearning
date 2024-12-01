# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# import Datasets
dataset = pd.read_csv("Social_Network_Ads.csv")
print(f"\nThe number of Rows, Columns are: \n{dataset.shape}\n")
x = dataset.iloc[:, :-1].values
'''print(f"\nColumns Except Last One: \n{x}\n")'''
y = dataset.iloc[:, -1].values
'print(f"\nThe Last column of the Dataset is: \n{y}\n")'

# Splitting our Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(f"\nLength of X_Train And X_Test: \n{len(x_train)} , {len(x_test)}\n")

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
'''print(f"\nScaled X_Train is: \n{x_train}\n")
print(f"\nScaled X_Test is: \n{x_test}\n")'''

# Creating a Model.
classifier = SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Prediction
print(classifier.predict(sc.transform([[32, 150000]]))[0])  # Take values from table to predict.
print(classifier.predict(sc.transform([[35, 108000]]))[0])
print(classifier.predict(sc.transform([[20, 86000]]))[0])
y_pred = classifier.predict(x_test)
print(y_pred)
'''#Gives [Predicted Values " " Original Values]'''
print(f"\nThe Predicted Values - Original Values of(Y_Pred, Y_Test):")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion Matrix And Accuracy
cm = confusion_matrix(y_test, y_pred)
print(f"\n Confusion Matrix is: {cm}\n")

# Accuracy Score How well Our prediction is working.
print(f"\nAccuracy Score: \n{accuracy_score(y_test, y_pred)}\n")
