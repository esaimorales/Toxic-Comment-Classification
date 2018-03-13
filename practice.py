import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

data = ['this is some text', 'this is not text', 'hello how are you', 'this code is clean']
train = ['some text is text', 'code is not clean', 'hello are you okay']

word_vectorizer = TfidfVectorizer(
    sublinear_tf = True,
    strip_accents = 'unicode',
    analyzer = 'word',
    token_pattern = r'\w{1,}',
    stop_words = 'english',
    ngram_range = (1, 1),
    max_features = 10
)

word_vectorizer.fit(data)
train_word_features = word_vectorizer.transform(train)
print word_vectorizer.get_feature_names()
a = train_word_features.toarray()
print a

print word_vectorizer.inverse_transform(a[1])
