# import Libraries
import numpy as np
import pandas as pd
from tensorflow.keras import datasets
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Here ML can't take images. so we convert images into three-dimensional matrix
# By using mnist dataset we can get pre converted matrix
'''print(y_train)
print(x_train)'''

'''print(x_train.shape)
# out: We are Having 60,000(images) and 28*28(width*height)
print(x_test.shape)
# out: We are Having 10,000(images) and same
'''
# Every image consist of 3 RGB(Red, Green, Blue) channels
# here in CNN(Convolutional Nural Network) it can't accept 28*28 format
# So we need to reshape our data elements into this -> (28*28*3) where 3(rgb value) represents the three color channels.

# Re-Shaping train and test data.
width, height = 28, 28
input_shape = (width, height, 1)
'''print(input_shape)'''
x_train = x_train.reshape(x_train.shape[0], height, width, 1)
x_test = x_test.reshape(x_test.shape[0], height, width, 1)

'''print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)'''

# Now Train, Test & Splitting our data using train_test_split
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
print(len(x_train))  # we are using 54,000 Training our data
print(len(x_val))  # we use(0.1) which is 6000 for Testing

# Normalizing Data
x_train = (x_train - x_train.mean())/ x_train.std()
x_val = (x_val - x_val.mean()) / x_val.std()
x_test = (x_test - x_test.mean()) / x_test.std()
'''print(x_train)
print(y_train)'''

# converting class labels into one-hot encoded vectors
from tensorflow import keras
num_labels = 10
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
y_val = keras.utils.to_categorical(y_val)
'''print(y_train)
print(y_test)
print(y_val)'''

# Building our Model:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, AveragePooling2D

# Refer to lenet5 Architecture
model = Sequential()

# Feature Extraction layers.
model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28, 28, 1)))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh'))

model.add(Flatten())
model.add(Dense(84, activation='tanh'))
model.add(Dense(num_labels, activation='softmax'))
'''print(model.summary())'''

# Compiling the Model
# For multiclass classification we use cross entropy.
# We use optimizer to Fasten up our training process ('adam' is the fastest optimizer)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, verbose=1)
print(f"\n Score(Loss, Accuracy):  \n{score}")