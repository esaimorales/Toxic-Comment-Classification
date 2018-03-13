import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

print 'reading data...'

train_data = pd.read_csv('train.csv').fillna(' ')
print len(train_data)

train = train_data[:100000]
test = train_data[100000:]

train_text = train['comment_text']
test_text = test['comment_text']
text = pd.concat([train_text, test_text])

print 'vectorizing...'

word_vectorizer = TfidfVectorizer(
    sublinear_tf = True,
    strip_accents = 'unicode',
    analyzer = 'word',
    token_pattern = r'\w{1,}',
    stop_words = 'english',
    ngram_range = (1, 1),
    max_features = 10000
)

print 'fitting data...'

word_vectorizer.fit(text)

train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

print train_word_features.toarray()
print train_word_features.toarray().shape

category_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

scores = []

for category_name in category_names:
    print 'classifiying', category_name

    test_target = test[category_name]
    classifier = LogisticRegression(C = 10, solver = 'sag')

    classifier.fit(test_word_features, test_target)
    x = classifier.predict_proba(test_word_features)[:,1]

    print x
    print len(x)
