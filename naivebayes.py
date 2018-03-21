import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('train.csv').fillna(' ')

# find data metrics



# dataset contains 159571 instances
train = train_data[:159000]
test = train_data[159000:]

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

train_vectorized = vectorizer_word.transform(train_text)
test_vectorized = vectorizer_word.transform(test_text)

char_vectorized = vectorizer_char.transform(train_text)
char_vectorized = vectorizer_char.transform(test_text)




# train_vectorized = vectorizer.fit_transform(train_text)
# print train_vectorized.toarray()
# print train_vectorized.toarray().shape
#
# test_vectorized = vectorizer.transform(test_text)
# print test_vectorized.toarray()
# print test_vectorized.toarray().shape

# vectorized_words = vectorizer_word.get_feature_names()
# print vectorized_words

model = MultinomialNB()
# model.fit(train_vectorized, train_y)

# prediction = model.predict(test_vectorized)

# print len(prediction), len(test_y)

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
prediction = []
scores = []

for category in categories:
    print 'predicting', category ,'...'
    model.fit(train_vectorized, train[category])
    prediction = model.predict(test_vectorized)
    scores.append(accuracy_score(test[category], prediction))
    print 'accuracy_score:', scores[-1]

print 'Average Accuracy:', np.mean(scores)
