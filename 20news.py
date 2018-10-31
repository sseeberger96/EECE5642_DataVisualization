from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn 
import string
import numpy as np

import gensim
from gensim.models.ldamodel import LdaModel




contractions = "n't'll're've's'm'd"

class LemmaTokenizer(object):
	def __init__(self):
		self.lem = WordNetLemmatizer()
	def __call__(self, doc):
		tokens = []

		for token in word_tokenize(doc):
			if len(wn.synsets(token)) == 0: 
				tokLem = self.lem.lemmatize(token)
				if string.punctuation.find(tokLem) == -1:
					if contractions.find(tokLem) == -1:
						tokens += [tokLem]
			else: 
				p = wn.synsets(token)[0].pos()
				tokLem = self.lem.lemmatize(token, pos=p)
				if string.punctuation.find(tokLem) == -1:
					if contractions.find(tokLem) == -1:
						tokens += [tokLem]
		return tokens


def preprocess(data, cat, PRINT): 
	countVect = CountVectorizer(stop_words = 'english', tokenizer=LemmaTokenizer())
	termMatrixSparse = countVect.fit_transform(data)
	
	tfidfTrans = TfidfTransformer()
	tfidfMatrix = tfidfTrans.fit_transform(termMatrixSparse)

	# Thresholding to get rid of low frequency words
	ignored_indices = tfidfMatrix < .1
	tfidfMatrix[ignored_indices] = 0

	termMatrix = termMatrixSparse.toarray()
	corp = []

	for doc in termMatrix:
		docWords = []
		index = 0
		for word in doc: 
			if word != 0: 
				docWords.append((index, word))
			index += 1
		corp.append(docWords)
	
	# print(tfidfMatrix.toarray())
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

	return corp, ignored_indices, countVect, tfidfMatrix

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
	
	print("category, docCount, sentCount, wordCount, numUniqueWords, meanSentLength, minSentLength, maxSentLength, stdSentLength")
	
	# Do all at once
	twentyNewsTrain = fetch_20newsgroups(subset='train', categories= ['sci.med'], shuffle=True, random_state=42)
	# twentyNewsTrain = fetch_20newsgroups(subset='train', categories= cats, shuffle=True, random_state=42)
	corpora, ignored_indices, processedVocab, processedWeights = preprocess(twentyNewsTrain.data[0:1], "sci.med", False)
	# stats = docStats(twentyNewsTrain.data, processedVocab, "total")

	# print(str(processedWeights.size))

	vocabDict = {};
	index = 0;
	for word in processedVocab.get_feature_names():
		vocabDict[index] = word;
		index = index + 1;

	print(str(vocabDict))
	print(corpora)
	lda = LdaModel(corpora, id2word=vocabDict, num_topics=5)
	

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
	





