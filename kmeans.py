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

	# input(sentences)
	'''
	# train model
	model = Word2Vec(sentences, min_count=1)
	# summarize the loaded model
	# print(model)
	# summarize vocabulary
	words = list(model.wv.vocab)
	# print(words)
	# access vector for one word
	# print(model['sentence'])
	# save model
	model.save('model.bin')
	'''
	return sentences

'''
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
'''


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

#create data frame that has the result of the MDS plus the cluster numbers and titles
print(len(xs))
print(len(cluster_colors.values()))
print(len(cluster_names.values()))
print(len(model.labels_.tolist()))

idx = 0;
X0 = []
Y0 = []
X1 = []
Y1 = []
X2 = []
Y2 = []
X3 = []
Y3 = []
for i in model.labels_.tolist():
	if model.labels_.tolist()[idx] == 0:
		X0.append(xs[idx])
		Y0.append(ys[idx])
	elif model.labels_.tolist()[idx] == 1:
		X1.append(xs[idx])
		Y1.append(ys[idx])
	elif model.labels_.tolist()[idx] == 2:
		X2.append(xs[idx])
		Y2.append(ys[idx])
	elif model.labels_.tolist()[idx] == 3:
		X3.append(xs[idx])
		Y3.append(ys[idx])

	idx = idx + 1;

trace0 = go.Scatter(
    x = X0,
    y = Y0,
    name = cluster_names[0],
    mode = 'markers',
    marker = dict(
        size = 10,
        color = cluster_colors[0],
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)'
        )
    )
)

trace1 = go.Scatter(
    x = X1,
    y = Y1,
    name = cluster_names[1],
    mode = 'markers',
    marker = dict(
        size = 10,
        color = cluster_colors[1],
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)'
        )
    )
)

trace2 = go.Scatter(
    x = X2,
    y = Y2,
    name = cluster_names[2],
    mode = 'markers',
    marker = dict(
        size = 10,
        color = cluster_colors[2],
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)'
        )
    )
)

trace3 = go.Scatter(
    x = X3,
    y = Y3,
    name = cluster_names[3],
    mode = 'markers',
    marker = dict(
        size = 10,
        color = cluster_colors[3],
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)'
        )
    )
)

data = [trace0, trace1, trace2, trace3]

layout = dict(title = 'Styled Scatter',
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False)
             )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='kmeans')


''' Plot with matplotlib
df = pd.DataFrame(dict(x=xs, y=ys, label=model.labels_.tolist(), title=model.labels_.tolist())) 

#group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

    
    
plt.show() #show the plot

#uncomment the below to save the plot if need be
# plt.savefig('clusters_small_noaxes.png', dpi=200)

'''
