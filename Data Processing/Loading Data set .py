#import Libraries to manage Data set

import numpy as np
import pandas as pd

#import Data Set
dataset = pd.read_csv('INTERN DATA.csv')
print(dataset)

#X takes all coloumbs from 0 - 8 except Last coloumb( :, : -1)
x = dataset.iloc[:, :-1].values
#y takes all Rows from but  except Last Row( :, : -1)
y = dataset.iloc[:, -1].values

'''print(x)
print("\n")
print(y)'''

#Taking Care of missing Data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, 1:3]) #x[All Rows, 1 to 3 coloumes]
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)

'''In this program first i have imported panda and numpys and then imported the 
Dataset Which is in the form of CSV and then 
1. Created two variables(X,Y) for representing rows and coloums
2. checked if x and y are giving data or not
3. imported SimpleIMputer from Sklearn
4. declared missing values and Strategy(MEAN) for imputer.
5. we fit the imputer to print dataSet with no missing values
6. we transfer x to Imputer.
7. printed the data set.
'''