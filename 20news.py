# General
import string
import numpy as np

# sklearn
from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# NLTK
from nltk import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn 
from nltk.corpus import stopwords

# Gensim
import gensim
from gensim.models.ldamodel import LdaModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
# %matplotlib inline

# Warnings
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)



contractions = ["n't", "'ll", "'re", "'ve", "'s", "'m", "'d", "\'s", "\'t"]

corpusSpecificStopwords = ['subject', '--']
tfidfThreshold = 0.1

class LemmaTokenizer(object):
	def __init__(self):
		self.lem = WordNetLemmatizer()
	def __call__(self, doc):
		tokens = []
		stopWords = set(stopwords.words('english')) 

		for token in regexp_tokenize(doc, pattern='\w+|\$[\d\.]+|\S+'):

			if string.punctuation.find(token) == -1: 
				if not token in contractions:
					if not token in stopWords: 
						if not token in corpusSpecificStopwords:
							if len(wn.synsets(token)) == 0: 
								tokLem = self.lem.lemmatize(token)
								tokens += [tokLem]
							else: 
								p = wn.synsets(token)[0].pos()
								tokLem = self.lem.lemmatize(token, pos=p)
								tokens += [tokLem]

		return tokens


def preprocess(data, cat, PRINT): 
	countVect = CountVectorizer(tokenizer=LemmaTokenizer())
	termMatrixSparse = countVect.fit_transform(data)
	
	tfidfTrans = TfidfTransformer(smooth_idf = False)
	tfidfMatrix = tfidfTrans.fit_transform(termMatrixSparse)

	# Thresholding to get rid of unimportant words
	meanWordWeights = np.mean(tfidfMatrix.A, axis=0)
	passingIndices = meanWordWeights > tfidfThreshold

	newVocabulary = np.array(countVect.get_feature_names())
	newVocabulary = newVocabulary[passingIndices]
	termMatrix = termMatrixSparse.toarray()
	termMatrix = termMatrix[:,passingIndices]

	vocabDict = {};
	index = 0;
	for word in newVocabulary:
		vocabDict[index] = word;
		index = index + 1;

	corp = []
	for doc in termMatrix:
		docWords = []
		index = 0
		for word in doc: 
			if word != 0: 
				docWords.append((index, word))
			index += 1
		corp.append(docWords)

	
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

	tokenize = LemmaTokenizer()
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
				minSentLength = numSentWords
				maxSentLength = numSentWords
				firstPass = 0
			elif numSentWords < minSentLength: 
				minSentLength = numSentWords
			elif numSentWords > maxSentLength: 
				maxSentLength = numSentWords
	catStat.close()
	meanSentLength = wordCount/sentCount
	varSentLength = (wordCountSquared/sentCount) - (meanSentLength**2)
	stdSentLength = np.float32(np.sqrt(varSentLength))
	numUniqueWords = len(vocab.get_feature_names())

	stats = [docCount, sentCount, wordCount, numUniqueWords, meanSentLength, minSentLength, maxSentLength, stdSentLength]
	
	print(str(cat) + ", " + str(stats).replace("[", '').replace("]", '') + str("\n"))
	return stats




if __name__ == '__main__':
	cats = ["comp.windows.x", "comp.os.ms-windows.misc", "talk.politics.misc", "comp.sys.ibm.pc.hardware","talk.religion.misc","rec.autos","sci.space","talk.politics.guns","alt.atheism","misc.forsale","comp.graphics","sci.electronics","sci.crypt","soc.religion.christian","rec.sport.hockey","sci.med","rec.motorcycles","comp.sys.mac.hardware","talk.politics.mideast","rec.sport.baseball"];
		
	# Do all at once
	twentyNewsTrain = fetch_20newsgroups(subset='train', categories= ['sci.med'], shuffle=True, random_state=42)
	# twentyNewsTrain = fetch_20newsgroups(subset='train', categories= cats, shuffle=True, random_state=42)
	corpora, vocabluary = preprocess(twentyNewsTrain.data[0:2], "sci.med", False)
	# stats = docStats(twentyNewsTrain.data, processedVocab, "total")

	# print(str(processedWeights.size))

	# print(str(vocabDict))
	# print(corpora)
	lda = LdaModel(corpora, id2word=vocabluary, num_topics=5)
	print(lda.print_topics())


	''' # Gather category specific document statistics
	f = open("./docStats.txt", 'w')
	f.write("category, docCount, sentCount, wordCount, numUniqueWords, meanSentLength, minSentLength, maxSentLength, stdSentLength\n")
	for entry in cats:
		
		twentyNewsTrain = fetch_20newsgroups(subset='train', categories= [entry], shuffle=True, random_state=42)
		ignored_indices, processedVocab, processedWeights = preprocess(twentyNewsTrain.data, entry, False)

		stats = docStats(twentyNewsTrain.data, processedVocab, f, entry)
		f.write(str(cat) + ", " + str(stats).replace("[", '').replace("]", '') + str("\n"))
	f.close()
	'''

	''' # Gather data for box-plot visualization
	# f = open("./docStats.txt", 'w')
	# f.write("category, docCount, sentCount, wordCount, numUniqueWords, meanSentLength, minSentLength, maxSentLength, stdSentLength\n")
	for entry in cats:
		
		twentyNewsTrain = fetch_20newsgroups(subset='train', categories= [entry], shuffle=True, random_state=42)
		ignored_indices, processedVocab, processedWeights = preprocess(twentyNewsTrain.data, False)

		stats = docStats(twentyNewsTrain.data, processedVocab, entry)
		# f.write(str(cat) + ", " + str(stats).replace("[", '').replace("]", '') + str("\n"))
	# f.close()
	'''
	





