import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('train.csv').fillna(' ')

print data

sns.pairplot(data = data)


train = data[:100000]
test = data[100000:]

train_text = train['comment_text']
test_text = test['comment_text']
all_text = data['comment_text']

# create TF-IDF word vectorizer
vectorizer_word = TfidfVectorizer(
    strip_accents = 'unicode',
    analyzer = 'word',
    stop_words = 'english',
    max_features = 10000
)

# create TF-IDF character vectorizer
vectorizer_char = TfidfVectorizer(
    strip_accents = 'unicode',
    analyzer = 'char',
    stop_words = 'english',
    max_features = 10000
)

vectorizer_word.fit(all_text)
vectorizer_char.fit(all_text)
