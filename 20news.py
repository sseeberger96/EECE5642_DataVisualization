# General
import string
import numpy as np

from scipy.cluster.hierarchy import ward, dendrogram

# sklearn
from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.manifold import MDS

# NLTK
from nltk import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn 
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

# Gensim
import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from gensim.test.utils import common_texts

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# sklearn stuff
from sklearn.decomposition import PCA

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.sklearn
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# %matplotlib inline

# Warnings
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import random
import pandas as pd

import os  # for os.path.basename

# import matplotlib.pyplot as plt
# import matplotlib as mpl

# PLotly is an alterative to matlibplot.
# conda install plotly
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

py.plotly.tools.set_credentials_file(username='tym0027', api_key='3z8kCvgxHDE7IS5gVblD')


contractions = ["n't", "'ll", "'re", "'ve", "'s", "'m", "'d", "\'s", "\'t", "\'ax"]

corpusSpecificStopwords = ['subject', '--', 'you', "\'\'", "``", "...", 'would', 'use', 'get', 'know', 'article', 'line', 'one', 'also', 'nntp-posting-host', 'reply-to', 
							'organization', 'wa', 'ha', 'write', 'could', 'doe', "\\/", 'hello', 'edu', 'cc', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
							'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ax', 'max', 'like', 'say', 'thanks', 'think', 'see', 'need', 'make', 
							'want', 'help', 'thing', 'come', 'may', 'many', 'people', 'time', 'go', 'take', 'well', 'even', 'good', 'right', 'much', 'way', 'year', 'try', 
							'work', 'going', 'really', 'seem', 'new', 'back', 'problem', 'two', 'look', 'mean', 'tell', 'sure', 'day', 'question', 'case', 'still', 'first'
							'please', 'give', 'maybe']
tfidfThreshold = 0

class customTokenizer(object):
	def __init__(self):
		self.lem = WordNetLemmatizer()
	def __call__(self, doc):
		tokens = customPreprocessor(self.lem, doc)
		return tokens


def customPreprocessor(lem, doc):
	tokens = []
	stopWords = set(stopwords.words('english')) 

	for token in regexp_tokenize(doc, pattern=r'\w+'):
		token = token.lower()
		if string.punctuation.find(token) == -1: 
			if not token in contractions:
				if not token.isdigit():
					if not wn.synsets(token): 
						pass
					else: 
						p = wn.synsets(token)[0].pos()
						tokLem = lem.lemmatize(token, pos=p)
						if not tokLem in stopWords: 
							if not tokLem in corpusSpecificStopwords:
								tokens += [tokLem]

	return tokens



def tryWord(word):
	if not wn.synsets(word):
		return False
	else:
		return True

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def preprocess(data, cat, PRINT): 
	countVect = CountVectorizer(tokenizer=customTokenizer())
	termMatrixSparse = countVect.fit_transform(data)

	lda = LatentDirichletAllocation(n_components=20, max_iter=15, learning_method='online',random_state=0).fit(termMatrixSparse)
	display_topics(lda, countVect.get_feature_names(), 10)

	vis = pyLDAvis.sklearn.prepare(lda, termMatrixSparse, countVect)
	pyLDAvis.save_html(vis, 'LDA_Visualization_sk.html')


	
	# tfidfTrans = TfidfTransformer(smooth_idf = False)
	# tfidfMatrix = tfidfTrans.fit_transform(termMatrixSparse)

	# # Thresholding to get rid of unimportant words
	# meanWordWeights = np.mean(tfidfMatrix.A, axis=0)
	# passingIndices = meanWordWeights > tfidfThreshold

	# newVocabulary = np.array(countVect.get_feature_names())
	# # print(newVocabulary)
	# newVocabulary = newVocabulary[passingIndices]
	# termMatrix = termMatrixSparse.toarray()
	# termMatrix = termMatrix[:,passingIndices]

	# vocabDict = {};
	# index = 0;
	# for word in newVocabulary:
	# 	vocabDict[index] = word;
	# 	index = index + 1;

	# corp = []
	# for doc in termMatrix:
	# 	docWords = []
	# 	index = 0
	# 	for word in doc: 
	# 		if word != 0: 
	# 			docWords.append((index, word))
	# 		index += 1
	# 	corp.append(docWords)

	
	if PRINT == True:
		catStat = open("preprocess_" + str(cat).replace(".","_") + ".txt", 'w')
		catStat.write(str("") + str("\n"))

		print(str(tfidfMatrix.toarray()))
		print(str(type(tfidfMatrix.toarray())))
		print(str(tfidfMatrix.toarray().size))

		index = 1;
		for entry in countVect.get_feature_names():
			catStat.write(str(entry) + str(", ") + str(tfidfMatrix.toarray().sum(axis=index)) + str("\n"))
			index = index + 1;
		catStat.close()

	return corp, vocabDict

def docStats(data, vocab, cat):
	docCount = 0
	sentCount = 0
	wordCount = 0 
	wordCountSquared = 0 
	firstPass = 1
	minSentLength = None
	maxSentLength = None

	tokenize = customTokenizer()
	catStat = open(str(cat).replace(".","_") + ".txt", 'w')
	catStat.write(str("data") + str("\n"))
	for doc in data: 
		docCount += 1
		for sentence in sent_tokenize(doc):
			# print("Sent --- " + sentence)
			sentCount += 1 
			numSentWords = len(tokenize(sentence))
			catStat.write(str(numSentWords) + str("\n"))
			wordCount += numSentWords
			wordCountSquared += numSentWords**2
			if firstPass:
				if  numSentWords == 0:
					continue

				minSentLength = numSentWords
				maxSentLength = numSentWords
				firstPass = 0
			elif numSentWords < minSentLength: 
				if  numSentWords == 0:
					continue

				minSentLength = numSentWords
			elif numSentWords > maxSentLength: 
				if  numSentWords == 0:
					continue

				maxSentLength = numSentWords
	catStat.close()
	meanSentLength = wordCount/sentCount
	varSentLength = (wordCountSquared/sentCount) - (meanSentLength**2)
	stdSentLength = np.float32(np.sqrt(varSentLength))
	# input("vocab: " + str(vocab))
	# numUniqueWords = len(vocab.get_feature_names())
	numUniqueWords = len(vocab)

	stats = [docCount, sentCount, wordCount, numUniqueWords, meanSentLength, minSentLength, maxSentLength, stdSentLength]
	
	print(str(cat) + ", " + str(stats).replace("[", '').replace("]", '') + str("\n"))
	return stats


def createDoc2VecModel(data):
	tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

	print("tagged_data: \n" + str(tagged_data))

	max_epochs = 100;
	vec_size = 20
	alpha = .025
	
	model = Doc2Vec(size=vec_size,
			alpha=alpha,
			min_alpha=.00025,
			min_count = 1,
			dm=1)

	model.build_vocab(tagged_data)

	for epoch in range(max_epochs):
		print('iteration {0}'.format(epoch))
		model.train(tagged_data,
			total_examples=model.corpus_count,
			epochs=model.iter)

		# decrease learning rate
		model.alpha -= .0002
		# fix the learning rate, no decay
		model.min_alpha = model.alpha
	model.save("d2v.model")
	print("Model saved...")

	return

def loadDoc2VecModel():
	return Doc2Vec.load('d2v.model.bak')

def createWord2VecModel(data):
	things = []
	sentences = []
	for doc in twentyNewsTrain.data:
		things.append(doc.split(". "))
	idx = 0;	
	for item in things:
		for doc in things[idx]:
			arr = []			
			word_arr = doc.split(" ")
			for word in word_arr:
				# input(word)
				if tryWord(word):
					#print(True)
					arr.append(word)
				else:
					continue
					#print(False)	
			sentences.append(arr)
		idx = idx + 1;

	# input(sentences)

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
	return

def displayWord2Vec(model):
	# fit a 2d PCA model to the vectors
	X = model[model.wv.vocab]
	pca = PCA(n_components=2)
	result = pca.fit_transform(X)
	# create a scatter plot of the projection
	plt.scatter(result[:, 0], result[:, 1])
	words = list(model.wv.vocab)
	for i, word in enumerate(words):
		plt.annotate(word, xy=(result[i, 0], result[i, 1]))
	plt.show()
	return

def makeLDA(data):

	fullVocab = []
	lem = WordNetLemmatizer()
	for doc in data: 
		docTokens = customPreprocessor(lem, doc)
		fullVocab.append(docTokens)

	docVocabDict = Dictionary(fullVocab)

	corp = [docVocabDict.doc2bow(text) for text in fullVocab]

	lda = LdaModel(corp, id2word=docVocabDict, num_topics=20, chunksize=100, random_state=100, update_every=1, alpha='auto', passes=10, per_word_topics=True)

	for thing in lda.print_topics():
		print(thing)

	print('\nPerplexity: ', lda.log_perplexity(corp))
	coherence_model_lda = CoherenceModel(model=lda, texts=fullVocab, dictionary=docVocabDict, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	print('\nCoherence Score: ', coherence_lda)


	vis = pyLDAvis.gensim.prepare(lda, corp, docVocabDict)
	pyLDAvis.save_html(vis, 'LDA_Visualization.html')

def createDocVocab(data):
	things = []
	sentences = []
	for doc in data:
		things.append(doc.split(". "))
	# print(things)

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

def kmeans(docs):
	cats = ["comp.windows.x", "comp.os.ms-windows.misc", "talk.politics.misc", "comp.sys.ibm.pc.hardware","talk.religion.misc","rec.autos","sci.space","talk.politics.guns","alt.atheism","misc.forsale","comp.graphics","sci.electronics","sci.crypt","soc.religion.christian","rec.sport.hockey","sci.med","rec.motorcycles","comp.sys.mac.hardware","talk.politics.mideast","rec.sport.baseball"];
	subcats = ["comp.windows.x", "sci.med", "rec.sport.hockey", "soc.religion.christian"]
	cats = ["comp.windows.x", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware","comp.graphics", "rec.autos","sci.space", "sci.med","talk.politics.misc", "talk.religion.misc", "alt.atheism", "soc.religion.christian", "rec.sport.hockey"]

	twentyNewsTrain = fetch_20newsgroups(subset='train', categories=cats, shuffle=True, random_state=42, remove=('headers'))

	# documents = createDocVocab(twentyNewsTrain.data[0:100])
	documents = docs;

	print(len(documents))

	vectorizer = TfidfVectorizer(stop_words='english')
	X = vectorizer.fit_transform(documents)

	true_k = 12
	model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10)
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
	mds = MDS(n_components=2, dissimilarity="precomputed", random_state=12)

	pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

	xs, ys = pos[:, 0], pos[:, 1]


	idx = 0;
	d = {}
	for i in range(0, true_k):
		d[i] = {"x":[], "y":[]}

	for i in model.labels_.tolist():
		d[model.labels_.tolist()[idx]]["x"].append(xs[idx])
		d[model.labels_.tolist()[idx]]["y"].append(ys[idx])
		idx = idx + 1;

	trace = []
	for i in range(0, true_k):
		trace.append(go.Scatter(x = d[i]["x"], y = d[i]["y"], name = cluster_names[i],mode = 'markers',marker = dict(size = 10, color = cluster_colors[i], line = dict(width = 2,color = 'rgb(0, 0, 0)'))));


	layout = dict(title = 'Styled Scatter', yaxis = dict(zeroline = False), xaxis = dict(zeroline = False))

	fig = dict(data=trace, layout=layout)
	plotly.offline.plot(fig, filename='kmeans')
	return

if __name__ == '__main__':
	# kmeans();

	cats = ["comp.windows.x", "comp.os.ms-windows.misc", "talk.politics.misc", "comp.sys.ibm.pc.hardware","talk.religion.misc","rec.autos","sci.space","talk.politics.guns","alt.atheism","misc.forsale","comp.graphics","sci.electronics","sci.crypt","soc.religion.christian","rec.sport.hockey","sci.med","rec.motorcycles","comp.sys.mac.hardware","talk.politics.mideast","rec.sport.baseball"];
	catsDescription = ["Alt - Atheism", "Computer - Graphics", "Computer - Microsoft/Windows", "Computer - IBM/PC", "Computer - Mac", "Computer - Windows", "Misc. - For Sale", "Rec. - Autos", "Rec - Motorcycles", "Sports - Baseball", "Sports - Hockey", "Science - Cryptography", "Science - Electronics", "Science - Medicine", "Science - Space", "Religion - Christianity", "Politics - Guns", "Politics - Middle East", "Politics - Misc.", "Religion - Misc."]
	subcats = ["comp.windows.x", "sci.med", "rec.sport.hockey", "soc.religion.christian"]
	
	''' # Gather category specific document statistics
	f = open("./docStats.txt", 'w')
	f.write("category, docCount, sentCount, wordCount, numUniqueWords, meanSentLength, minSentLength, maxSentLength, stdSentLength\n")
	for entry in cats:
		
		catStats = fetch_20newsgroups(subset='train', categories= [entry], shuffle=True, random_state=42)
		corp, vocab = preprocess(catStats.data, entry, False)

		stats = docStats(catStats.data, vocab, entry)
		f.write(str(entry) + ", " + str(stats).replace("[", '').replace("]", '') + str("\n"))
		print(str(entry) + ", " + str(stats).replace("[", '').replace("]", '') + str("\n"))
	'''
	# Do all at once
	twentyNewsTrain = fetch_20newsgroups(subset='train', categories= cats, shuffle=True, random_state=42, remove=('headers'))
	# print(twentyNewsTrain.target[0:5])

	fullVocab = []
	lem = WordNetLemmatizer()
	kmeansDocs = []

	idx = 0
	for doc in twentyNewsTrain.data: 
		# print(float(idx / len(twentyNewsTrain.data) * 100))
		docTokens = customPreprocessor(lem, doc)
		fullVocab.append(docTokens)
		kmeansDocs.append(" ".join(docTokens))
		idx = idx + 1;

	# input("doc_tokens: \n" + str(docTokens))
	
	# input("kmeans vocab: \n" + str(kmeansDocs))
	# kmeans(kmeansDocs[0:100]);

	docVocabDict = Dictionary(fullVocab)

	corp = [docVocabDict.doc2bow(text) for text in fullVocab]

	documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(fullVocab)]
	model = Doc2Vec(documents, vector_size=20, min_count=1, workers=4)
	# print(model.docvecs.vectors_docs)

	X = model.docvecs.vectors_docs
	pca = PCA(n_components=2)
	result = pca.fit_transform(X)
	# create a scatter plot of the projection 
	print(twentyNewsTrain.target[0:20])
	print(twentyNewsTrain.filenames[0:20])
	for i in range(20):
		indices = [j for j, x in enumerate(twentyNewsTrain.target) if x == i]
		plt.scatter(result[indices, 0], result[indices, 1], label=catsDescription[i])
	# words = list(model.wv.vocab)
	# for i, word in enumerate(words):
	# 	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
	plt.legend()
	plt.show()

	kmeans_model = KMeans(n_clusters=20, init='k-means++', max_iter=100)
	X = kmeans_model.fit(model.docvecs.doctag_syn0)
	labels=kmeans_model.labels_.tolist()

	l = kmeans_model.fit_predict(model.docvecs.doctag_syn0)
	pca = PCA(n_components=2).fit(model.docvecs.doctag_syn0)
	datapoint = pca.transform(model.docvecs.doctag_syn0)
	plt.figure
	plt.scatter(datapoint[:, 0], datapoint[:, 1], c=labels)
	plt.show()



	# corpora, vocabulary = preprocess(twentyNewsTrain.data, subcats, False)
	# stats = docStats(twentyNewsTrain.data, vocabulary, "total")
	
	# # f.write(str("total") + ", " + str(stats).replace("[", '').replace("]", '') + str("\n"))
	# print(str("total") + ", " + str(stats).replace("[", '').replace("]", '') + str("\n"))
	# # f.close()

	# LDA Work
	# vocabByDoc = []
	# for doc in corpora:
	# 	wordsInDoc = []
	# 	# print(doc)
	# 	for word in doc: 
	# 		for freq in range(word[1]):
	# 			wordsInDoc.append(vocabulary[word[0]])
	# 	vocabByDoc.append(wordsInDoc)
	# # print(vocabByDoc)
	# docVocabDict = Dictionary(vocabByDoc)
	# print(docVocabDict)
	
	# newCorp = [docVocabDict.doc2bow(text) for text in vocabByDoc]

	# makeLDA(twentyNewsTrain.data)
	
	
	

	# Doc 2 Vec work
	# createDoc2VecModel(twentyNewsTrain.data[0:5])
	# doc2VecModel = loadDoc2VecModel();

	''' Using Doc 2 Vec model
	sims = doc2VecModel.docvecs.most_similar(99)

	print(sims)

	print(doc2VecModel.doesnt_match("halloween costume devil party  scarf".split()))
	print(doc2VecModel.doesnt_match("black blue yellow shirt navy black green orange".split()))
	print(doc2VecModel.doesnt_match("summer winter fall t-shirt spring hot cold".split()))
	print(doc2VecModel.doesnt_match("straight slim fit custom regular winter".split()))


	print(doc2VecModel.most_similar(positive=['boy', 'king'], negative=['girl']))
	print(doc2VecModel.most_similar(positive=['blue', 'shirt'], negative=['blue']))
	print(doc2VecModel.most_similar(positive=['calvin', 'klein'], negative=['tommy']))
	'''
	

	# # word 2 vec work
	# # createWord2VecModel(twentyNewsTrain.data);
	
	# # load model
	# model = Word2Vec.load('model.bin')
	# # print(model)

	# # displayWord2Vec(model);
	# # input("press any key to continue...")

	# # keyed vector work
	# # from: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/ 
	# filename = "./model.bin"
	# m = KeyedVectors.load(filename)

	# while True:
	# 	try:
	# 		pos = input("positive: ").split(" ")
	# 		# neg = input("negative: ").split(" ")
	# 		result = m.most_similar(pos, topn=5)
	# 		print(result)
	# 	except KeyError:
	# 		print("Not in vocab...")
	# 		continue
	
	




