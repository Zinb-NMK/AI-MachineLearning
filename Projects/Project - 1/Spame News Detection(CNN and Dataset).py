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
import requests
from bs4 import BeautifulSoup

# Load Datasets
True_news = pd.read_csv('True.csv')
Fake_news = pd.read_csv('Fake.csv')

# Add label to deside whether text is(T/F)
True_news['label'] = 0
Fake_news['label'] = 1

# Select only required DataElements.
dataset1 = True_news[['text', 'label']]
dataset2 = Fake_news[['text', 'label']]
dataset = pd.concat([dataset1, dataset2])
'''print(f"\nNo of Rows, No of Columns: \n{dataset.shape}\n")
print(f"\nCheck for Null Values in Dataset: \n{dataset.isnull().sum()}\n")'''

# To check no of True and False DataElement Count.
'''print(f"\nNo of True and False Values: \n{dataset['label'].value_counts()}\n")'''

# For Shuffling Data Elements(To Avoid bias and help models learn better).
dataset = dataset.sample(frac=1)
'''print(f"\nShuffled Dataset: \n{dataset}\n")
'''
# NLP
nltk.download('stopwords')
nltk.download('wordnet')
ps = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Define Clean_row to clean
def clean_row(row):
    row = row.lower()
    row = re.sub('[^a-zA-Z]', ' ', row)  # Replace non-alphabetic Characters with space
    token = row.split()

    # Lemmatizing and removing stopwords
    news = [ps.lemmatize(word) for word in token if word not in stop_words]
    cleanned_news = ' '.join(news)
    return cleanned_news


# Clean the text from data from CSV files.
dataset['text'] = dataset['text'].apply(lambda x: clean_row(x))
'''print(dataset['text'])'''

# TfidfVectorizer is used to convert data to (0's and 1's).
vectorizer = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1, 2))
x = dataset.iloc[:35000, 0]  # 35000 data elements are for training and rest for testing.
y = dataset.iloc[:35000, 1]  # y(Label) - X(data)
'''print(x)
print(y)'''

# Splitting data into train and test
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=0)

# fit and transform train data
vec_train_data = vectorizer.fit_transform(train_data)
'''print(f"\n Unique tokens: \n{vec_train_data.shape[1]}\n")
'''
vec_train_data = vec_train_data.toarray()
'''print(f"\n Type of Vectorizer: \n{type(vec_train_data)}\n")
'''

# fit and transform testing Data
vec_test_data = vectorizer.transform(test_data)
vec_test_data = vec_test_data.toarray()

# To find Shape of the data.
'''print(f"\n Shape of(Train, Test) is: \n{vec_train_data.shape}, {vec_test_data.shape}")
print(f"\nEncoded Train Data into(0's and 1's): \n{vec_train_data}\n")
'''
# Creating DataFrames from the vectorized data.
train_data_df = pd.DataFrame(vec_train_data, columns=vectorizer.get_feature_names_out())
testing_data_df = pd.DataFrame(vec_test_data, columns=vectorizer.get_feature_names_out())
'''print(f"Final Train Data is: \n{train_data_df}\n")
'''
# MODEL (Naive Bayes Classifier.)
clf = MultinomialNB()
clf.fit(train_data_df, train_label)

# predict on test and train data
y_pred_test = clf.predict(testing_data_df)
y_pred_train = clf.predict(train_data_df)
'''print(f"\n Test Label After implementing Multinomial Model is: \n{test_label}\n")
print(f"\n Predicted Value(Y-Pred): \n{y_pred_test}\n")
'''

# Accuracy
print(f"\n The Accuracy of Testing Label is: \n{accuracy_score(test_label, y_pred_test)}\n")
print(f"\n The Accuracy of Training Label is: \n{accuracy_score(train_label, y_pred_train)}\n")


# New: Function to scrape latest news articles from a website (CNN)
def scrape_latest_news(url):
    response = requests.get(url)  # fetching the web content.
    soup = BeautifulSoup(response.text, 'html.parser')  # html parsing.

    articles = soup.find_all('h3', class_='title')
    # Based on website structure adjust selector.

    news_list = []
    for article in articles:
        title = article.get_text()  # Extract text from article tags.
        news_list.append(title)

    return news_list


# Scraping Function
latest_news_url = 'https://www.cnn.com/'
latest_articles = scrape_latest_news(latest_news_url)

# Scraping article for prediction.
for article in latest_articles:
    cleaned_article = clean_row(article)
    news_vectorized = vectorizer.transform([cleaned_article])  # Transform scraped article.

    # Scraped news is true or false.
    predicted_label = clf.predict(news_vectorized.toarray())
    if predicted_label == 0:
        print(f"News: '{article}' is True.")
    else:
        print(f"News: '{article}' is Fake.")
else:
    print("No articles found.")

# my inputs for given true and fake.csv files.
print("Enter News items one by one (type 'done' when completed): ")
while True:
    txt = input("Enter News:- ")
    if txt.strip().lower() == 'done':
        break  # if user want to finish.

    txt_cleaned = clean_row(str(txt))
    pred = clf.predict(vectorizer.transform([txt_cleaned]).toarray())
    if pred == 0:
        print("News is True.")
    else:
        print("News is fake.")
