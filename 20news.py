from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn 
import string
import numpy as np

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


def preprocess(data): 
	countVect = CountVectorizer(stop_words = 'english', tokenizer=LemmaTokenizer())
	termMatrix = countVect.fit_transform(data)
	# print(countVect.get_stop_words())
	# print(countVect.get_feature_names())
	# print(termMatrix.toarray())
	tfidfTrans = TfidfTransformer()
	tfidfMatrix = tfidfTrans.fit_transform(termMatrix)
	# print(tfidfMatrix.toarray())
	# print(countVect.get_feature_names()[34])
	return countVect, tfidfMatrix

def docStats(data, vocab):
	docCount = 0
	sentCount = 0
	wordCount = 0 
	wordCountSquared = 0 
	firstPass = 1
	minSentLength = None
	maxSentLength = None

	tokenize = LemmaTokenizer()

	for doc in data: 
		docCount += 1
		for sentence in sent_tokenize(doc):
			# print("Sent --- " + sentence)
			sentCount += 1 
			numSentWords = len(tokenize(sentence))
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

	meanSentLength = wordCount/sentCount
	varSentLength = (wordCountSquared/sentCount) - (meanSentLength**2)
	stdSentLength = np.float32(np.sqrt(varSentLength))
	numUniqueWords = len(vocab.get_feature_names())

	stats = [docCount, sentCount, wordCount, numUniqueWords, meanSentLength, minSentLength, maxSentLength, stdSentLength]
	# print(stats)
	return stats




if __name__ == '__main__':
	twentyNewsTrain = fetch_20newsgroups(subset='train', categories= ['sci.med'], shuffle=True, random_state=42)
	processedVocab, processedWeights = preprocess(twentyNewsTrain.data[0:2])
	print(processedVocab.get_feature_names())
	stats = docStats(twentyNewsTrain.data[0:2], processedVocab)
	print(stats)






