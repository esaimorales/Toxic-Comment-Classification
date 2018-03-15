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
test_y = test['toxic']

# print train_y

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

for i, value in enumerate(train['toxic']):
    if value == 1:
        print train['comment_text'][i]

vectorizer = TfidfVectorizer(
    strip_accents = 'unicode',
    stop_words = 'english',
    max_features = 10000
)

train_vectorized = vectorizer.fit_transform(train_text)
# print train_vectorized.toarray()
# print train_vectorized.toarray().shape

test_vectorized = vectorizer.transform(test_text)
# print test_vectorized.toarray()
# print test_vectorized.toarray().shape

vectorized_words = vectorizer.get_feature_names()
print vectorized_words

model = LogisticRegression()
model.fit(train_vectorized, train_y)

prediction = model.predict(test_vectorized)

print len(prediction), len(test_y)
print test_y

# count = 0
# for i, value in enumerate(prediction):
#     if value == test_y[i]:
#         count += 1

# print 'hit:', count
# print 'miss:', len(prediction) - count
# print float(count)/float(len(prediction))


print 'ACCURACY:', accuracy_score(test_y, prediction)

#verify

# print train_text
# print prediction[:30]

# 2 features
# MNB   0.904618018835
# LR    0.904618018835

# 5 features
# MNB   0.904618018835
# LR    0.904618018835

# 1000 features
# MNB   0.944620704705
# LR    0.949488845243

# 10000 features
# MNB   0.948330563529
# LR    0.955464907421
