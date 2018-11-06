from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer

from scipy.cluster.hierarchy import ward, dendrogram

from nltk.corpus import wordnet as wn
import random
import pandas as pd

import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

py.plotly.tools.set_credentials_file(username='tym0027', api_key='3z8kCvgxHDE7IS5gVblD')

# see: http://brandonrose.org/clustering#K-means-clustering 

def tryWord(word):
	if not wn.synsets(word):
		return False
	else:
		return True


def createDocVocab(data):
	things = []
	sentences = []
	for doc in data:
		things.append(doc.split(". "))
		
	idx = 0;	
	for item in things:
		for doc in things[idx]:
			arr = []			
			word_arr = doc.split(" ")
			for word in word_arr:
				if tryWord(word):
					arr.append(word)
				else:
					continue
			sentences.append(" ".join(arr))
		idx = idx + 1;

	return sentences

cats = ["comp.windows.x", "comp.os.ms-windows.misc", "talk.politics.misc", "comp.sys.ibm.pc.hardware","talk.religion.misc","rec.autos","sci.space","talk.politics.guns","alt.atheism","misc.forsale","comp.graphics","sci.electronics","sci.crypt","soc.religion.christian","rec.sport.hockey","sci.med","rec.motorcycles","comp.sys.mac.hardware","talk.politics.mideast","rec.sport.baseball"];
subcats = ["comp.windows.x", "sci.med", "rec.sport.hockey", "soc.religion.christian"]
twentyNewsTrain = fetch_20newsgroups(subset='train', categories=subcats, shuffle=True, random_state=42, remove=('headers'))

documents = createDocVocab(twentyNewsTrain.data[0:100])

print(len(documents))

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 4
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
cluster_names = {};
cluster_colors = {};
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']
for i in range(true_k):
    print("\nCluster " + str(i) + ":");
    t = str(i) + ") ";
    j = 0;
    for ind in order_centroids[i, :10]:
        print("\t" + str(terms[ind]))
        if j < 4:
            t = t + str(terms[ind]) + str(" ")
            j = j + 1;

    cluster_names[i] = t
    cluster_colors[i] = colors[i]
    print

#set up colors per clusters using a dict
# cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
 	      
print(cluster_names)
print(cluster_colors)

# vectorizer = HashingVectorizer(n_features=2**4)
# Xs = vectorizer.fit_transform(documents)
# print(Xs.shape)

# tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses
print(X.shape)
dist = 1 - cosine_similarity(X)


MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]


idx = 0;
d = {}
for i in range(0, len(subcats)):
	d[i] = {"x":[], "y":[]}

for i in model.labels_.tolist():
	d[model.labels_.tolist()[idx]]["x"].append(xs[idx])
	d[model.labels_.tolist()[idx]]["y"].append(ys[idx])
	idx = idx + 1;

trace = []
for i in range(0, len(subcats)):
	trace.append(go.Scatter(x = d[i]["x"], y = d[i]["y"], name = cluster_names[i],mode = 'markers',marker = dict(size = 10, color = cluster_colors[i], line = dict(width = 2,color = 'rgb(0, 0, 0)'))));


layout = dict(title = 'Styled Scatter', yaxis = dict(zeroline = False), xaxis = dict(zeroline = False))

fig = dict(data=trace, layout=layout)
plotly.offline.plot(fig, filename='kmeans')

