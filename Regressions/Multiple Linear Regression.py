import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# import DataSets
dataset = pd.read_csv("company_data.csv")
print(f"\nThe DataSet is: \n{dataset}\n")

# We use x-variable to store first 4 columns(R&D to State)
# y-Variable store Profit column.
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(f"\nRepresent first 4 columns: \n {x}\n")
print(f"\nRepresent last columns: \n {y}\n")

# Encoding Categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(f"\n Encoded Categorical Values are: \n {x}\n")

# Data Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"\nTrain Variable(X): \n{x_train}\n")
print(f"\nTrain Variable(Y): \n{y_train}\n")
print(f"\nLength of train Model: \n{len(y_train)}\n")
# Gives the length of train or test values

# Creating Model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Predicting
y_pred = reg.predict(x_test)
np.set_printoptions(precision=2)  # make predict values upto 2points
print(f"\nThe Predicted Values and original Values(PREDICTED - ORIGINAL - Y):")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

'''
1. import Dataset.
2. Encode Data Categorically.
3. Split Data AS done in previous linear regression project.
4. Creat Linear Model.
5. predicting data'''