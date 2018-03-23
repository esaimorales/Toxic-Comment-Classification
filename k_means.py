from sklearn import cluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD



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
    stop_words = 'english',
    max_features = 2
)

# vectorize text
vectorizer_word.fit(all_text)

# transform text to vectors
X_train = vectorizer_word.transform(train_text)
X_test = vectorizer_word.transform(test_text)

# a = np.asarray(X_train)
np.savetxt("foo.csv", X_train, delimiter=",")

X_train = X_train.toarray()

print X_train
print X_train[0]
print X_train[1]



# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# lda = LDA(n_components=2) #2-dimensional LDA
# X_t = pd.DataFrame(lda.fit_transform(X_train.toarray(), train['toxic']))
# X_t = lda.fit(X_train.toarray(), train['toxic']).transform(X_train.toarray())

# y = train['toxic']

# plt.scatter(lda_transformed[y==0][0], lda_transformed[y==0][1], label='Class 1', c='red')
# plt.scatter(lda_transformed[y==1][0], lda_transformed[y==1][1], label='Class 2', c='blue')

# Display legend and show plot
# plt.legend(loc=3)
# plt.show()
# target_names = ['non_toxic', 'toxic']
# colors = ['navy', 'turquoise']
#
# y = train['toxic']
#
# print X_t
#
# print X_t[0]
# print X_t[1]

# plt.figure()
#
# # for color, i, target_name in zip(colors, [0, 1], target_names):
# plt.scatter(X_train[0], X_train[1], alpha=.8, color='navy', label='non_toxic')
# # plt.scatter(X_t[1], X_t[1], alpha=.8, color='turquoise', label='toxic')
#
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('Title')
#
# plt.show()

# print X_train.toarray()
#
# svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
# X_train_svd = svd.fit_transform(np.array(X_train))
#
# print X_train_svd.toarray()

# print X_train

# X_train.todense()
# X_test.todense()

# pca = PCA(n_components=2).fit(X_train)

# k = 2
# kmeans = cluster.KMeans(n_clusters=k)
# kmeans.fit(X_train)
#
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
#
# from matplotlib import pyplot
#
# for i in range(k):
#     # select only data observations with cluster label == i
#     ds = data[np.where(labels==i)]
#     # plot the data observations
#     pyplot.plot(ds[:,0],ds[:,1],'o')
#     # plot the centroids
#     lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
#     # make the centroid x's bigger
#     pyplot.setp(lines,ms=15.0)
#     pyplot.setp(lines,mew=2.0)
#
# pyplot.show()
