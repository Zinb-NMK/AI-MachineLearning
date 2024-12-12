import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# import Datasets
dataset = pd.read_csv("Social_Network_Ads.csv")
print(f"\nThe number of Rows, Columns are: \n{dataset.shape}\n")

# Splitting our Dataset
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Training Dataset.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(f"\nLength of X_Train And X_Test: \n{len(x_train)} , {len(x_test)}\n")

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Creating a Model.
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Prediction
y_pred = classifier.predict(x_test)
print(f"\n Predicted Values of (Y-Pred) are: \n{y_pred}\n")

# Confusion Matrix And Accuracy
cm = confusion_matrix(y_test, y_pred)
print(f"\n Confusion Matrix is: \n {cm}\n")

# Accuracy Score
print(f"\nAccuracy Score: \n{accuracy_score(y_test, y_pred)}\n")

# Visualizing the Logistic Regression results (Training Set)
plt.figure(figsize=(10, 6))

# Plotting decision boundary
x_set, y_set = x_train, y_train  # Use x_train and y_train for training data visualization
X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

plt.contourf(X1, X2, Z, alpha=0.75, cmap='Blues')
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plotting the points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], label=f'Class {j}', edgecolor='k', s=50)

plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
