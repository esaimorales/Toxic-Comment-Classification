import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

train_data = pd.read_csv('train.csv').fillna(' ')

train = train_data[:100000]
test = train_data[100000:]

train_y = train['toxic']
test_y = train['toxic']

print train_y

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

vectorizer = TfidfVectorizer(
    strip_accents = 'unicode',
    stop_words = 'english',
    max_features = 10
)

train_vectorized = vectorizer.fit_transform(train_text)
print train_vectorized.toarray()
print train_vectorized.toarray().shape

test_vectorized = vectorizer.transform(test_text)
print test_vectorized.toarray()
print test_vectorized.toarray().shape

print vectorizer.get_feature_names()


model = MultinomialNB()
model.fit(train_vectorized, train_y)

prediction = model.predict(test_vectorized)

count = 0
for i, value in enumerate(prediction):
    if value == test_y[i]:
        count += 1

print 'hit:', count
print 'miss:', len(prediction) - count
print float(count)/float(len(prediction))
