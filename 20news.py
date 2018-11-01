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
from gensim.models import CoherenceModel
from gensim.test.utils import common_texts

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



contractions = ["n't", "'ll", "'re", "'ve", "'s", "'m", "'d", "\'s", "\'t", "\'ax"]

corpusSpecificStopwords = ['subject', '--', 'you', "\'\'", "``", "...", 'would', 'use', 'get', 'know', 'article', 'line', 'one', 'also', 'nntp-posting-host', 'reply-to', 
							'organization', 'wa', 'ha', 'write', 'could', 'doe', "\\/", 'hello', 'edu', 'cc', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
							'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
tfidfThreshold = 0

class LemmaTokenizer(object):
	def __init__(self):
		self.lem = WordNetLemmatizer()
	def __call__(self, doc):
		tokens = []
		stopWords = set(stopwords.words('english')) 

		for token in regexp_tokenize(doc, pattern=r'\w+'):
			if string.punctuation.find(token) == -1: 
				if not token in contractions:
					if not token.isdigit():
						if len(wn.synsets(token)) == 0: 
							'''
							tokLem = self.lem.lemmatize(token)
							if not tokLem in stopWords: 
								if not tokLem in corpusSpecificStopwords:
									tokens += [tokLem]
							'''
							pass
						else: 
							p = wn.synsets(token)[0].pos()
							tokLem = self.lem.lemmatize(token, pos=p)
							if not tokLem in stopWords: 
								if not tokLem in corpusSpecificStopwords:
									tokens += [tokLem]

		return tokens



def tryWord(word):
	if not wordnet.synsets(word):
		return False
	else:
		return True


def preprocess(data, cat, PRINT): 
	countVect = CountVectorizer(tokenizer=LemmaTokenizer())
	termMatrixSparse = countVect.fit_transform(data)
	
	tfidfTrans = TfidfTransformer(smooth_idf = False)
	tfidfMatrix = tfidfTrans.fit_transform(termMatrixSparse)

	# Thresholding to get rid of unimportant words
	meanWordWeights = np.mean(tfidfMatrix.A, axis=0)
	passingIndices = meanWordWeights > tfidfThreshold

	newVocabulary = np.array(countVect.get_feature_names())
	# print(newVocabulary)
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
	subcats = ["comp.windows.x", "sci.med", "rec.sport.hockey", "soc.religion.christian"]
	# Do all at once
	twentyNewsTrain = fetch_20newsgroups(subset='train', categories= subcats, shuffle=True, random_state=42, remove=('headers'))
	# twentyNewsTrain = fetch_20newsgroups(subset='train', categories= cats, shuffle=True, random_state=42)
	corpora, vocabulary = preprocess(twentyNewsTrain.data, "sci.med", False)
	# stats = docStats(twentyNewsTrain.data, processedVocab, "total")

	# print(str(processedWeights.size))

	# print(str(vocabDict))
	#  print(corpora)
	lda = LdaModel(corpora, id2word=vocabulary, num_topics=4, chunksize=100, random_state=100, update_every=1, alpha='auto', passes=5, per_word_topics=True)

	for thing in lda.print_topics():
		print(thing)
		print("\n")

	print('\nPerplexity: ', lda.log_perplexity(corpora))
	coherence_model_lda = CoherenceModel(model=lda, corpus=corpora, dictionary=vocabulary, coherence='u_mass')
	coherence_lda = coherence_model_lda.get_coherence()
	print('\nCoherence Score: ', coherence_lda)


	# vis = pyLDAvis.gensim.prepare(lda, corpora, vocabulary)
	# pyLDAvis.save_html(vis, 'LDA_Visualization.html')


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
	





