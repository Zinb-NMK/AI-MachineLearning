# Import Libraries
import pandas as pd
import vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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
print(f"\nNo of Rows, No of Columns: \n{dataset.shape}\n")
print(f"\nCheck for Null Values in Dataset: \n{dataset.isnull().sum()}\n")
# To check no of True and False DataElement Count.
print(f"\nNo of True and False Values: \n{dataset['label'].value_counts()}\n")
# For Shuffling Data Elements(To Avoid bias and help models learn better).
dataset = dataset.sample(frac=1)
'''print(f"\nShuffled Dataset: \n{dataset}\n")
'''
# NLP
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
ps = WordNetLemmatizer()
stopwords = stopwords.words('english')


def clean_row(row):
    row = row.lower()
    row = re.sub('[^a-zA-Z]', '', row)
    token = row.split()

    # Lemmatizing
    news = [ps.lemmatize(word) for word in token if not word in stopwords]
    cleanned_news = ' '.join(news)
    return cleanned_news


dataset['text'] = dataset['text'].apply(lambda x : clean_row(x))
'''print(dataset['text'])'''

# TfidfVectorizer is used to convert data to (0's and 1's).
vectorizer = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1, 2))
x = dataset.iloc[:35000, 0]  # 35000 data elements are for training and rest for testing.
y = dataset.iloc[:35000, 1]  # y(Label) - X(data)
'''print(x)
print(y)'''
from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=0)

vec_train_data = vectorizer.fit_transform(train_data)
print(f"\n Unique tokens: \n{vec_train_data.shape[1]}\n")
# Training Data
vec_train_data = vec_train_data.toarray()
print(f"\n Type of Vectorizer: \n{type(vec_train_data)}\n")

# Testing Data
vec_test_data = vectorizer.transform(test_data)
vec_test_data = vec_test_data.toarray()
# To find Shape of the data.
print(f"\n Shape of(Train, Test) is: \n{vec_train_data.shape}, {vec_test_data.shape}")
print(f"\nEncoded Train Data into(0's and 1's): \n{vec_train_data}\n")

# Creating DataFrames from the vectorized data.
train_data_df = pd.DataFrame(vec_train_data, columns=vectorizer.get_feature_names_out())
testing_data_df = pd.DataFrame(vec_test_data, columns=vectorizer.get_feature_names_out())
print(f"Final Train Data is: \n{train_data_df}\n")

# MODEL
from sklearn.naive_bayes import multinomialNB


