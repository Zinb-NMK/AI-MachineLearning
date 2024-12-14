import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
from flask import Flask, render_template, request

app = Flask(__name__)

# File paths
true_news_path = r'./Dataset/True_News.csv'
fake_news_path = r'./Dataset/Fake_News.csv'

# Check if the files exist
if not os.path.exists(true_news_path) or not os.path.exists(fake_news_path):
    print("Error: One or more dataset files not found!")
    exit()

# Load datasets
true_news = pd.read_csv(true_news_path, encoding='ISO-8859-1', low_memory=False)
fake_news = pd.read_csv(fake_news_path, encoding='ISO-8859-1', low_memory=False)

# Verify required columns exist
if 'text' not in true_news.columns or 'text' not in fake_news.columns:
    print("Error: Missing 'text' column in dataset.")
    exit()

# Label the datasets (True = 0, Fake = 1)
true_news['label'] = 0
fake_news['label'] = 1

# Combine datasets
dataset1 = true_news[['text', 'label']].dropna()
dataset2 = fake_news[['text', 'label']].dropna()
dataset = pd.concat([dataset1, dataset2])

# Handle missing values
dataset['text'] = dataset['text'].fillna('')

# Data Cleaning Function
nltk.download('stopwords')
stopwords_list = stopwords.words('english')


def clean_row(row):
    if isinstance(row, str):  # Check if the value is a string
        row = row.lower()  # Convert to lowercase
        row = re.sub('[^a-zA-Z\\s]', '', row)  # Remove special chars and numbers
        return row
    return ""


# Apply the cleaning function to the text data
dataset['text'] = dataset['text'].apply(lambda x: clean_row(x))

# Convert text data into numerical features using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, lowercase=True, ngram_range=(1, 1))
x = dataset['text']
y = dataset['label']

# Train-Test Split
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=0)

# Vectorize Data
vec_train_data = vectorizer.fit_transform(train_data)
vec_test_data = vectorizer.transform(test_data)

# Train the Multinomial Naive Bayes Model
clf = MultinomialNB()
clf.fit(vec_train_data, train_label)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(clf, vectorizer.transform(x), y, cv=5)  # 5-fold cross-validation
print(f"Cross-validation accuracy: {cv_scores.mean() * 100:.2f}%")

# Predict on the test set
test_predictions = clf.predict(vec_test_data)
accuracy = accuracy_score(test_label, test_predictions)
print(f"Model Accuracy on test data: {accuracy * 100:.2f}%")

# Print confusion matrix and classification report for detailed analysis
print("Confusion Matrix:")
print(confusion_matrix(test_label, test_predictions))
print("\nClassification Report:")
print(classification_report(test_label, test_predictions))


# Route for the home page
@app.route('/')
def index():
    steps = [
        "Step 1: Import necessary libraries and modules (e.g., pandas, sklearn, nltk, etc.)",
        "Step 2: Load and preprocess the dataset, checking if the files exist.",
        "Step 3: Label the datasets (True = 0, Fake = 1) and combine them into one dataset.",
        "Step 4: Handle missing values in the text data.",
        "Step 5: Clean the data by removing special characters and lowercasing.",
        "Step 6: Convert the text data into numerical features using TfidfVectorizer.",
        "Step 7: Split the data into training and testing sets (80%-20%).",
        "Step 8: Vectorize the training and testing data.",
        "Step 9: Train a Multinomial Naive Bayes model on the training data.",
        "Step 10: Evaluate the model on the test data and print accuracy.",
        "Step 11: Set up continuous input for predictions (e.g., via the web interface)."
    ]
    return render_template('index.html', steps=steps)


# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    txt = request.form['news']
    if not txt.strip():
        result = "Error: Please enter valid news text."
    else:
        news = clean_row(txt)
        pred = clf.predict(vectorizer.transform([news]))
        result = "News is Correct (True)" if pred[0] == 0 else "News is Fake"

    return render_template('index.html', result=result, steps=[
        "Step 1: Import necessary libraries and modules (e.g., pandas, sklearn, nltk, etc.)",
        "Step 2: Load and preprocess the dataset, checking if the files exist.",
        "Step 3: Label the datasets (True = 0, Fake = 1) and combine them into one dataset.",
        "Step 4: Handle missing values in the text data.",
        "Step 5: Clean the data by removing special characters and lowercasing.",
        "Step 6: Convert the text data into numerical features using TfidfVectorizer.",
        "Step 7: Split the data into training and testing sets (80%-20%).",
        "Step 8: Vectorize the training and testing data.",
        "Step 9: Train a Multinomial Naive Bayes model on the training data.",
        "Step 10: Evaluate the model on the test data and print accuracy.",
        "Step 11: Set up continuous input for predictions (e.g., via the web interface)."
    ])


if __name__ == "__main__":
    app.run(debug=True)
