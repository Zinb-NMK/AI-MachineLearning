# import Libraries.
import numpy as np
import pandas as pd

# Load DataSet
dataset = pd.read_csv('Mall_Customers.csv')
print(f"\nThe DataSet is: \n{dataset}\n")
x = dataset.iloc[:, [3, 4]].values
y = dataset.iloc[:, -1].values
print(x)

# ELBOW METHOD
from sklearn.cluster import KMeans
wcss = []
for i in range(2, 51):  # For cluster, it should be minimum 2. so we stated with 2.
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print(wcss)

# Matplotlib is used to plot the graph. It forms an elbow like structure.
import matplotlib.pyplot as plt
plt.plot(range(2, 51), wcss)
print(plt.show())  # Shows the graph
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_means = kmeans.fit_predict(x)
print(f"\nClustered Representation of Data(y_means): \n{y_means}\n")

# Visualizing
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s=100, c='red', label='cluster 1')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s=100, c='blue', label='cluster 2')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s=100, c='green', label='cluster 3')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s=100, c='yellow', label='cluster 4')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s=100, c='cyan', label='cluster 5')
print(f"\n Scatter Cluster Data is(Visible on plot area.): \n{plt.show()}\n")

'''K-MEAN CLUSTERING:- 
1.Import libraries to get data like(numpy & pandas).
2.Load Dataset to assign X & Y values.
3.Import K-Means(Elbow Model) from sklearn.cluster.
4.Assign Operational range for Cluster Method.
5.Use Matplotlib to plot the scatter Data and Elbow line(Can Be Seen On Plot Area).
'''