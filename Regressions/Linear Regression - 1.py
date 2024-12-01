import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import dataset
dataset = pd.read_csv('YearsExperience_Salary.csv')
print(f"\nDataSet Values are: \n{dataset.head(10)}\n")   # Print upto 10 values from dataset

x = dataset.iloc[:, :-1].values  # gives Experience
y = dataset.iloc[:, -1].values  # gives salary

print(f"\nThe Years of Experience: \n {x}\n")
print(f"\nThe Salary of all Employs is: \n {y}\n")

'''#import train_test_split.'''
# Splitting dataset.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Creating our model(linear Model.)
reg = LinearRegression()
reg.fit(x_train, y_train)

# Predict
print(f"\nPassing Values from Data set:\n {x_test}\n  ")
y_pred = reg.predict(x_test)  # Used to predict based on given values.
print(f"\n Predicted values based on X_train Values: \n{y_pred}\n")
'''Check difference from y_pred and y_test'''
print(f"\n y_test(salary): \n{y_test}\n")

# Visualizing training set.
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, reg.predict(x_train), color='blue')
plt.show()
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, reg.predict(x_test), color='blue')
plt.show()  # Command to get visual info


'''In this prog first we imported dataset and processed the data available in the
given data set using train_test_split.
1. we import linear Regression from sklearn for predicting salary as above.
2. the we import matplotlib to visualize data
3. we use scatter and plot to visualize data.
4. we use plot.show() to show our visualized data'''