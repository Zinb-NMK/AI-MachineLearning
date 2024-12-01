# import Libraries to manage Data set
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# import Data Set
dataset = pd.read_csv('INTERN DATA.csv')
print(dataset)

# X takes all columns from 0 to 8 except Last column( :, : -1)
x = dataset.iloc[:, :-1].values
# y takes all Rows from but  except Last Row( :, : -1)
y = dataset.iloc[:, -1].values

# Just to check weather we are getting desired output.
'''print(x)
print("\n")
print(y)'''

# Taking Care of missing Data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])  # x[All Rows, 1 to 3 columns]
x[:, 1:3] = imputer.transform(x[:, 1:3])
print("\n Updated Values: \n", x)

'''In this program first i have imported panda and numpy and then imported the 
Dataset Which is in the form of CSV and then 
1. Created two variables(X,Y) for representing rows and columns.
2. checked if x and y are giving data or not
3. imported SimpleImputer from Sklearn
4. declared missing values and Strategy(MEAN) for imputer.
5. we fit the imputer to print dataSet with no missing values
6. we transfer x to Imputer.
7. printed the data set.
'''