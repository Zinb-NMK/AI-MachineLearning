import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# import DataSets
dataset = pd.read_csv("company_data.csv")
print(f"\nThe DataSet is: \n{dataset}\n")

# We use x-variable to store first 4 columns (R&D to State)
# y-Variable store Profit column.
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(f"\nRepresent first 4 columns: \n{x}\n")
print(f"\nRepresent last column (Profit): \n{y}\n")

# Encoding Categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(f"\nEncoded Categorical Values are: \n{x}\n")

# Data Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"\nTrain Variable(X): \n{x_train}\n")
print(f"\nTrain Variable(Y): \n{y_train}\n")

# Creating Model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Predicting
y_pred = reg.predict(x_test)
np.set_printoptions(precision=2)  # Make predicted values up to 2 points
print(f"\nThe Predicted Values and Original Values (Predicted - Actual):")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Plotting the results: Predicted vs Actual (2D plot)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Identity line (y=x)
plt.title('Multiple Linear Regression: Predicted vs Actual')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.show()
