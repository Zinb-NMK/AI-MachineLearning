# Categorical Values can only accept Numeric values."
# Encoding Categorical Data

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# import Data Set
dataset = pd.read_csv('INTERN DATA.csv')
print(f"\nAll Elements in Dataset are: \n{dataset}\n")

# X takes all columns from 0 to 8 except Last columns( :, : -1)
x = dataset.iloc[:, :-1].values
# y takes all Rows from but  except Last Row( :, : -1)
y = dataset.iloc[:, -1].values

# Taking Care of missing Data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])  # x[All Rows, 1 to 3 columns]
x[:, 1:3] = imputer.transform(x[:, 1:3])
print("\n Updated Values: \n", x)

'''(Independent Variable.)'''
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print("\nAfter Encoding Independent Variable: \n", x)

'''(Dependent Variable.)'''
le = LabelEncoder()
y = le.fit_transform(y)
print("\nAfter Encoding Dependent Variable: \n", y)


'''In this program we first find missing values as Loading Data Set program. and then 
1. we imported the sklearn sklearn.compose import ColumnTransformer
2. we imported sklearn.preprocessing import OneHotEncoder.
3. First By using encoder and OneHotEncoder. We Encoded country names as 0.0 & 1.0 (as in output).  
4. Then we encoder labelEncoder to encode the Yes and No As "0 & 1" as in output and the output will be in Array .
 '''
