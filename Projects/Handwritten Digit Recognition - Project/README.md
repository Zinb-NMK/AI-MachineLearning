<pre>
<h1>MNIST Digit Classification with CNN</h1>
This project uses a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.

<h2>📋 Requirements</h2>
Python 3.x
TensorFlow
Keras
NumPy
scikit-learn


🧑‍💻 Project Steps

1. Import Libraries: Import necessary libraries like TensorFlow, Keras, and scikit-learn.
Load Dataset: The MNIST dataset is loaded using Keras.

2. Preprocess Data:
Reshape images for CNN input (28x28x1 format).
Normalize pixel values for faster convergence.
One-hot encode the labels (digits 0-9).

3. Model Architecture:
A CNN model inspired by LeNet-5 with 3 convolutional layers and pooling.
Fully connected layers for classification.

4. Train the Model:
Split the data into training, validation, and test sets.
Compile and train the model.
Evaluate the Model: Test the model on the test set to measure performance.


📊 Conclusion
The CNN model achieves high accuracy in classifying handwritten digits from the MNIST dataset. 
Further improvements can be made by tuning the model architecture or exploring different hyperparameters.










<</pre>
