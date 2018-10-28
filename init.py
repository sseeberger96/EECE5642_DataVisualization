from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn 
import string

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
	print(countVect.get_feature_names())
	# print(termMatrix.toarray())






if __name__ == '__main__':
	twentyNewsTrain = fetch_20newsgroups(subset='train', categories= ['sci.med'], shuffle=True, random_state=42)
	preprocess(twentyNewsTrain.data[0:2])






