import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack


# read training data
print 'reading data...'
train_data = pd.read_csv('train.csv').fillna(' ')
test_data = pd.read_csv('test.csv').fillna(' ')

# extract comment text from data
train_text = train_data['comment_text']
test_text = test_data['comment_text']

# print train_data['comment_text'][1]
# print '---------------------------'
for i, value in enumerate(train_data['identity_hate']):
    if value == 1:
        print train_data['comment_text'][i]

# concatenate all text
all_text = pd.concat([train_text, test_text])

# create word vectorizer
print 'vectorizing...'
word_vectorizer = TfidfVectorizer(
    sublinear_tf = True,
    strip_accents = 'unicode',
    analyzer = 'word',
    token_pattern = r'\w{1,}',
    stop_words = 'english',
    ngram_range = (1, 1),
    max_features = 10000)

# fit data
print 'fitting data...'
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

# set column headers
category_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

scores = []
submission = pd.DataFrame.from_dict({'id': test_data['id']})

for category_name in category_names:
    print 'classifiying', category_name

    train_target = train_data[category_name]
    classifier = LogisticRegression(C=10, solver='sag')
    cv_score = np.mean(cross_val_score(classifier, train_word_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)

    print 'CV score for class {} is {}'.format(category_name, cv_score)
    classifier.fit(train_word_features, train_target)
    submission[category_name] = classifier.predict_proba(test_word_features)[:, 1]

# create submission
print 'creating submission'

print 'Total CV score is {}'.format(np.mean(scores))
submission.to_csv('submission.csv', index=False)

print 'done!'
