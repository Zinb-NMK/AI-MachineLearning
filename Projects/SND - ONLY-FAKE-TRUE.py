# Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Datasets
True_news = pd.read_csv('True.csv')
Fake_news = pd.read_csv('Fake.csv')

# Add label to decide whether text is True (0) or Fake (1)
True_news['label'] = 0
Fake_news['label'] = 1

# Select only required DataElements and concatenate datasets
dataset = pd.concat([True_news[['text', 'label']], Fake_news[['text', 'label']]])
dataset = dataset.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset

# NLP Preparation
nltk.download('stopwords')
nltk.download('wordnet')

ps = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_row(row):
    row = row.lower()
    row = re.sub('[^a-zA-Z]', ' ', row)  # Replace non-alphabetic characters with space
    tokens = row.split()
    # Lemmatizing and removing stopwords
    cleaned_news = ' '.join([ps.lemmatize(word) for word in tokens if word not in stop_words])
    return cleaned_news


# Clean the text data
dataset['text'] = dataset['text'].apply(clean_row)

# Vectorization using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1, 2))
x = dataset['text']
y = dataset['label']

# Split the data into training and testing sets
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=0)

# Fit and transform the training data
vec_train_data = vectorizer.fit_transform(train_data)

# Transform the testing data
vec_test_data = vectorizer.transform(test_data)

# Create DataFrames from the vectorized data (optional)
train_data_df = pd.DataFrame(vec_train_data.toarray(), columns=vectorizer.get_feature_names_out())
testing_data_df = pd.DataFrame(vec_test_data.toarray(), columns=vectorizer.get_feature_names_out())

# MODEL: Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(train_data_df, train_label)

# Predictions on test and training data
y_pred_test = clf.predict(testing_data_df)
y_pred_train = clf.predict(train_data_df)

# Accuracy Evaluation
print(f"\nThe Accuracy of Testing Label is: {accuracy_score(test_label, y_pred_test)}")
print(f"\nThe Accuracy of Training Label is: {accuracy_score(train_label, y_pred_train)}")

# User Input for Prediction
txt = input("Enter News: ")
news_cleaned = clean_row(txt)

# Transforming news input into a format compatible with the model (as a DataFrame)
news_vectorized = vectorizer.transform([news_cleaned])
news_df = pd.DataFrame(news_vectorized.toarray(), columns=vectorizer.get_feature_names_out())

# Making predictions
predicted_label = clf.predict(news_df)

if predicted_label == 0:
    print("News is True.")
else:
    print("News is Fake.")
