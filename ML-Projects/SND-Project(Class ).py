# import libraries.
import numpy as np
import pandas as pd

true_news = pd.read_csv('True_News.csv')
fake_news = pd.read_csv('Fake_News.csv')
'''print(true_news)
print(fake_news)'''

# Label is a separate column that we create to assign a True or False Value(0, 1)
true_news['label'] = 0
fake_news['label'] = 1

'''print(true_news)
print(fake_news)'''
dataset1 = true_news[['text', 'label']]
dataset2 = fake_news[['text', 'label']]

# This dataset will have two data points they are (Text(From all datasets), Label(0, 1)).
dataset = pd.concat([dataset1, dataset2])
''' print(dataset.shape)'''

# To find dataset has any null points we use
''' print(dataset.isnull().sum())'''

# to find how many labels are in 0, 1 are
''' print(dataset['label'].value_counts())'''
# used to shuffle dataset.
dataset = dataset.sample(frac=1)
''' print(dataset)'''

# NLP(Natural language processing )
import nltk
# re works on fuzzy logics and helps in data cleaning.
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ps = WordNetLemmatizer()
stopwords = stopwords.words('english')
nltk.download('wordnet')

# Creating a function for cleaning a Row


def clean_row(row):
    row = row.lower()
    row = re.sub('[^a-zA-Z]', ' ', row)  # this line replace all the nums and special with space(" ")
    # split
    token = row.split()

    news = [ps.lemmatize(word) for word in token if word not in stopwords]

    cleanned_news = " ".join(news)
    return cleanned_news


''' print(dataset['text'])'''
dataset['text'] = dataset['text'].apply(lambda x: clean_row(x))  # clean data using function clean_row(x)
'''print(dataset['text'])'''
from sklearn.feature_extraction.text import TfidfVectorizer  #
vectorizer = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1, 2))
x = dataset.iloc[:35000, 0]
y = dataset.iloc[:35000, 1]
'''print(x)
print(y)
'''
# Import splitting functions
from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=0)
'''print(train_data)
print(test_data)'''

# vectorized training data
vec_train_data = vectorizer.fit_transform(train_data).toarray()

# vectorized testing data.
vec_test_data = vectorizer.transform(test_data).toarray()
'''print(vec_train_data)
print(vec_train_data.shape, vec_test_data.shape)
'''
training_data = pd.DataFrame(vec_train_data, columns=vectorizer.get_feature_names_out())
testing_data = pd.DataFrame(vec_test_data, columns=vectorizer.get_feature_names_out())
'''print(testing_data)
print(training_data)'''

# Model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(vec_train_data, train_label)
y_pred_test = clf.predict(vec_test_data)
y_pred_train = clf.predict(vec_train_data)
'''print(y_pred)
print(test_label)
for accuracy'''

from sklearn.metrics import accuracy_score
print(f"\n The accuracy of testing data is: \n{accuracy_score(test_label, y_pred_test)}\n")
print(f"\n The accuracy of training data is: \n{accuracy_score(train_label, y_pred_train)}\n")

# Continuous input for predictions.
while True:
    txt = input("Enter News(or type 'exit' to quit): ")
    if txt.lower() == 'exit':
        print("Exiting the program.")
        break
    # cleaning input text and make predictions.
    news = clean_row(str(txt))
    pred = clf.predict(vectorizer.transform([news]).toarray())

    if pred == 0:
        print("News is Correct")
    else:
        print("News is Fake")

