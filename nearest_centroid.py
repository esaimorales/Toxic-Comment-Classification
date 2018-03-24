from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import decomposition


""" Implements Nearest Centroid Model on Toxic Comments Dataset """

import numpy as np
import pandas as pd

data = pd.read_csv('train.csv').fillna(' ')

train = data[:100000]
test = data[100000:]

# extract text
train_text = train['comment_text']
test_text = test['comment_text']
all_text = data['comment_text']

# create TF-IDF word vectorizer
vectorizer_word = TfidfVectorizer(
    strip_accents = 'unicode',
    analyzer = 'word',
    stop_words = 'english'
    max_features = 10000,
)

# vectorize text
vectorizer_word.fit(all_text)

# transform text to vectors
X_train = vectorizer_word.transform(train_text)
X_test = vectorizer_word.transform(test_text)

# pca = decomposition.SparsePCA(n_components = 10)
# pca.fit(X_train.toarray())
# X_train_pca = pca.transform(X_train)
# print X_train_pca

nc = NearestCentroid()

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
scores = []

for category in categories:

    nc.fit(X_train, train[category])

    print 'predicting', category

    print 'train data:'
    prediction = nc.predict(X_train)
    print accuracy_score(train[category], prediction)

    print 'test data:'
    prediction = nc.predict(X_test)
    scores.append(accuracy_score(test[category], prediction))
    print scores[-1]

print 'average accuracy: ', np.mean(scores)
