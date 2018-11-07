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
from sklearn.manifold import TSNE

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
	# countVect = CountVectorizer(tokenizer=customTokenizer()) 
	countVect = CountVectorizer(tokenizer=customTokenizer(), max_features=2000, max_df=0.5, min_df=2)
	termMatrix = countVect.fit_transform(data)

	tfidfTrans = TfidfTransformer(smooth_idf = False)
	tfidfMatrix = tfidfTrans.fit_transform(termMatrix)
	
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

	return countVect, termMatrix, tfidfMatrix 

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

def createWord2VecModel(data):
	rawSentences = []
	sentences = []
	lem = WordNetLemmatizer()
	for doc in twentyNewsTrain.data:
		rawSentences.append(sent_tokenize(doc))
	idx = 0;	
	for item in rawSentences:
		for sent in item:
			docTokens = customPreprocessor(lem, sent)
			sentences.append(docTokens)

		# for doc in things[idx]:
		# 	arr = []			
		# 	word_arr = doc.split(" ")
		# 	for word in word_arr:
		# 		# input(word)
		# 		if tryWord(word):
		# 			#print(True)
		# 			arr.append(word)
		# 		else:
		# 			continue
		# 			#print(False)	
		# 	sentences.append(arr)
		# idx = idx + 1;

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

def displayWord2Vec(model, wordsToPlot):
	# fit a 2d PCA model to the vectors

	X = model[wordsToPlot]
	pca = PCA(n_components=2)
	result = pca.fit_transform(X)
	# create a scatter plot of the projection
	plt.scatter(result[:, 0], result[:, 1])
	words = list(wordsToPlot)
	for i, word in enumerate(words):
		plt.annotate(word, xy=(result[i, 0], result[i, 1]))
	plt.title('Labeled Vector Representations of Words - Small Subset\n(Word2Vec)')
	plt.axis('off')
	plt.show()
	return

def makeLDA(countVect, termMatrix):
	lda = LatentDirichletAllocation(n_components=20, max_iter=15, learning_method='online',random_state=0).fit(termMatrix)
	display_topics(lda, countVect.get_feature_names(), 10)

	vis = pyLDAvis.sklearn.prepare(lda, termMatrix, countVect)
	pyLDAvis.save_html(vis, 'LDA_Visualization_sk.html')

def makeDoc2Vec(fullVocab):
	documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(fullVocab)]
	model = Doc2Vec(documents, vector_size=20, min_count=1, workers=4)
	# print(model.docvecs.vectors_docs)

	X = model.docvecs.vectors_docs
	pca = PCA(n_components=2)
	result = pca.fit_transform(X)
	# create a scatter plot of the projection 
	# print(twentyNewsTrain.target[0:20])
	# print(twentyNewsTrain.filenames[0:20])
	for i in range(20):
		indices = [j for j, x in enumerate(twentyNewsTrain.target) if x == i]
		plt.scatter(result[indices, 0], result[indices, 1], label=catsDescription[i])

	leg = plt.legend(frameon=False)
	leg.set_draggable(True)
	plt.title('Labeled Vector Representations of Documents\n(Doc2Vec)')
	plt.axis('off')
	plt.show()

	return model


def kMeansDocVec(doc2VecModel, nClasses):
	kmeans_model = KMeans(n_clusters=nClasses, init='k-means++', max_iter=100)
	X = kmeans_model.fit(doc2VecModel.docvecs.doctag_syn0)
	labels=kmeans_model.labels_.tolist()

	pca = PCA(n_components=2).fit(doc2VecModel.docvecs.doctag_syn0)
	datapoint = pca.transform(doc2VecModel.docvecs.doctag_syn0)
	plt.figure
	plt.scatter(datapoint[:, 0], datapoint[:, 1], c=labels)
	plt.show()

def kMeansTFIDF(tfidfMatrix, target, nClasses):
	kmeans_model = KMeans(n_clusters=nClasses, init='k-means++', max_iter=100, n_init=1)
	X = kmeans_model.fit(tfidfMatrix)
	labels=kmeans_model.labels_.tolist()

	pca = PCA(n_components=2).fit(tfidfMatrix.A)
	datapoint = pca.transform(tfidfMatrix.A)
	
	plt.figure
	# for i in range(nClasses):
	
	indices = [j for j, x in enumerate(target) if x == 0]
	plt.scatter(datapoint[indices, 0], datapoint[indices, 1], c='#440154ff', label=subcatsDescription[0])

	indices = [j for j, x in enumerate(target) if x == 1]
	plt.scatter(datapoint[indices, 0], datapoint[indices, 1], c='#365c8cff',  label=subcatsDescription[1])

	indices = [j for j, x in enumerate(target) if x == 3]
	plt.scatter(datapoint[indices, 0], datapoint[indices, 1], c= '#238b8dff', label=subcatsDescription[3])

	indices = [j for j, x in enumerate(target) if x == 2]
	plt.scatter(datapoint[indices, 0], datapoint[indices, 1], c= '#55c467ff', label=subcatsDescription[2])
	
	leg = plt.legend(loc=8, frameon=False, mode='expand', ncol=nClasses)
	leg.set_draggable(True)
	plt.axis('off')
	plt.title('Ground Truth of TF-IDF Vectors\n(4 Category Subset)')
	plt.show()



	plt.figure

	indices = [j for j, x in enumerate(labels) if x == 0]
	plt.scatter(datapoint[indices, 0], datapoint[indices, 1], c='#440154ff')

	indices = [j for j, x in enumerate(labels) if x == 1]
	plt.scatter(datapoint[indices, 0], datapoint[indices, 1], c='#365c8cff')

	indices = [j for j, x in enumerate(labels) if x == 3]
	plt.scatter(datapoint[indices, 0], datapoint[indices, 1], c= '#238b8dff')

	indices = [j for j, x in enumerate(labels) if x == 2]
	plt.scatter(datapoint[indices, 0], datapoint[indices, 1], c= '#55c467ff')
	
	plt.axis('off')
	plt.title('K-Means Clustering of TF-IDF Vectors\n(4 Category Subset)')
	plt.show()

	# plt.scatter(datapoint[:, 0], datapoint[:, 1], c=labels)
	# plt.axis('off')
	# plt.show()


if __name__ == '__main__':

	cats = ["comp.windows.x", "comp.os.ms-windows.misc", "talk.politics.misc", "comp.sys.ibm.pc.hardware","talk.religion.misc","rec.autos","sci.space","talk.politics.guns","alt.atheism","misc.forsale","comp.graphics","sci.electronics","sci.crypt","soc.religion.christian","rec.sport.hockey","sci.med","rec.motorcycles","comp.sys.mac.hardware","talk.politics.mideast","rec.sport.baseball"];
	catsDescription = ["Alt - Atheism", "Computer - Graphics", "Computer - Microsoft/Windows", "Computer - IBM/PC", "Computer - Mac", "Computer - Windows", "Misc. - For Sale", "Rec. - Autos", "Rec - Motorcycles", "Sports - Baseball", "Sports - Hockey", "Science - Cryptography", "Science - Electronics", "Science - Medicine", "Science - Space", "Religion - Christianity", "Politics - Guns", "Politics - Middle East", "Politics - Misc.", "Religion - Misc."]
	subcats = ["comp.windows.x", "sci.med", "rec.sport.hockey", "soc.religion.christian"]
	subcatsDescription = ["Computer - Windows", "Sports - Hockey", "Science - Medicine", "Religion - Christianity"]
	
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
	twentyNewsTrain = fetch_20newsgroups(subset='train', categories=cats, shuffle=True, random_state=42, remove=('headers'))

	
	# fullVocab = []
	# lem = WordNetLemmatizer()
	# for doc in twentyNewsTrain.data: 
	# 	docTokens = customPreprocessor(lem, doc)
	# 	fullVocab.append(docTokens)

	# docVocabDict = Dictionary(fullVocab)
	# corp = [docVocabDict.doc2bow(text) for text in fullVocab]

	
	countVect, termMatrix, tfidfMatrix = preprocess(twentyNewsTrain.data, subcats, False)
	
	# kMeans tfidf
	# kMeansTFIDF(tfidfMatrix, twentyNewsTrain.target, 4)

	# LDA
	# makeLDA(countVect, termMatrix)

	# doc2Vec
	# doc2VecModel = makeDoc2Vec(fullVocab)

	# kMeans doc2Vec
	# kMeansDocVec(doc2VecModel, 20)






	# stats = docStats(twentyNewsTrain.data, vocabulary, "total")
	
	# # f.write(str("total") + ", " + str(stats).replace("[", '').replace("]", '') + str("\n"))
	# print(str("total") + ", " + str(stats).replace("[", '').replace("]", '') + str("\n"))
	# # f.close()


	
	
	

	# # word 2 vec work
	# createWord2VecModel(twentyNewsTrain.data);
	
	# load model
	model = Word2Vec.load('model.bin')
	# print(model)

	wordsToPlot = np.array(countVect.get_feature_names())
	displayWord2Vec(model, wordsToPlot);
	# input("press any key to continue...")

	# keyed vector work
	# from: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/ 
	filename = "./model.bin"
	m = KeyedVectors.load(filename)

	while True:
		try:
			pos = input("positive: ").split(" ")
			# neg = input("negative: ").split(" ")
			result = m.most_similar(pos, topn=5)
			print(result)
		except KeyError:
			print("Not in vocab...")
			continue
	
	




