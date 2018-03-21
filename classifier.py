import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# parse dataset file
train_data = pd.read_csv('train.csv').fillna(' ')

# dataset contains 159571 instances
train = train_data[:100000]
test = train_data[100000:]

# extract text
train_text = train['comment_text']
test_text = test['comment_text']
all_text = train_data['comment_text']

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

# vectorize text
vectorizer_word.fit(all_text)
vectorizer_char.fit(all_text)

# vectorize words
train_vectorized_words = vectorizer_word.transform(train_text)
test_vectorized_words = vectorizer_word.transform(test_text)

# vectorize characters
train_vectorized_chars = vectorizer_char.transform(train_text)
test_vectorized_chars = vectorizer_char.transform(test_text)

# combine feature set with vectorized words & characters
train_features = hstack([train_vectorized_words, train_vectorized_chars])
test_features = hstack([test_vectorized_words, test_vectorized_chars])

# create models
lr = LogisticRegression()
nb = MultinomialNB()

models = [lr, nb]
model_names = ['Logistic Regression', 'Naive Bayes']

# set categories
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
scores = []

for i, model in enumerate(models):
    print model_names[i]

    # make predictions for each category
    for category in categories:
        print 'predicting {}...'.format(category)
        model.fit(train_features, train[category])
        # make prediction
        prediction = model.predict(test_features)
        # calculate accuracy
        scores.append(accuracy_score(test[category], prediction))
        print 'accuracy score:', scores[-1]

    print 'Average Accuracy:', np.mean(scores)
    print '-----------------------------'
