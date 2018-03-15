import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('train.csv').fillna(' ')

train = train_data[:100000]
test = train_data[100000:]

train_y = train['toxic']
test_y = test['identity_hate']

# print train_y

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

# for i, value in enumerate(train['toxic']):
#     if value == 1:
#         print train['comment_text'][i]

vectorizer = TfidfVectorizer(
    strip_accents = 'unicode',
    stop_words = 'english',
    max_features = 1000
)

vectorizer.fit(all_text)
train_vectorized = vectorizer.transform(train_text)
# print train_vectorized.toarray()

test_vectorized = vectorizer.transform(test_text)
# print test_vectorized.toarray()

# train_vectorized = vectorizer.fit_transform(train_text)
# print train_vectorized.toarray()
# print train_vectorized.toarray().shape
#
# test_vectorized = vectorizer.transform(test_text)
# print test_vectorized.toarray()
# print test_vectorized.toarray().shape

vectorized_words = vectorizer.get_feature_names()
# print vectorized_words

model = MultinomialNB()
# model.fit(train_vectorized, train_y)

# prediction = model.predict(test_vectorized)

# print len(prediction), len(test_y)

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
prediction = []

for category in categories:
    print 'predicting', category ,'...'
    model.fit(train_vectorized, train[category])
    prediction = model.predict(test_vectorized)
    print 'accuracy_score:', accuracy_score(test[category], prediction)

# print 'ACCURACY:', accuracy_score(test_y, prediction)

#verify
# for i, row in enumerate(test_text):
#     if prediction[i] == 1:
#         print 'text:', row
#         print 'prediction:', prediction[i]
#         print 'actual:', np.array(test_y)[i]
