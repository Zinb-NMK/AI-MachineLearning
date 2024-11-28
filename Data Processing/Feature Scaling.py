import numpy as np
import pandas as pd

#import Data Set
dataset = pd.read_csv('INTERN DATA.csv')
print(dataset)

#X takes all column from 0 to 8 except Last column( :, : -1)
x = dataset.iloc[:, :-1].values
#y takes all Rows from but  except Last Row( :, : -1)
y = dataset.iloc[:, -1].values

#Taking Care of missing Data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, 1:3]) #x[All Rows, 1 to 3 columns]
x[:, 1:3] = imputer.transform(x[:, 1:3])
print("\n Updated Values: \n",x)


#Encoding Categorical Data

'''(Independent Variable.)'''

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [0])], remainder= 'passthrough')


x = np.array(ct.fit_transform(x))
print("\nAfter Encoding Independent Variable: \n",x)

#Data Set Split.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)#testing_size vary based on dataset quantity we require less testing.
print(f"\n After training with random State Value(Training Set : X_train): \n",x_train)

print(f"\n After testing with random State Value(Testing Set : X_test): \n",x_test)

print(f"\n After training with State Value(Targeted Training Set : y_train): \n",y_train)

print(f"\n After testing with State Value(Targeted Testing Set : y_test): \n",y_test)

'''
1. we import train_test_split from sklearn.
2. here we will define x_train, x_test, y_train, y_test
3. x train and test will have 5 columns values 
4. y train and test will have targeted values on which we are working with
5. train_test_split is used to split the data set.
6. test_size = 0.2 means we need 20% of data as testing set and remaining as training set.
 '''


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.fit_transform(x_test[:, 3:])

print(f"\n After Scaling  with StandardScaler: \n {x_train}")

print(f"\n After Scaling with StandardScaler: \n {x_test}")

'''By using feature scaling we can equally prioritize all elements in the data set'''