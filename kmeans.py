from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import fetch_20newsgroups 
from nltk.corpus import wordnet as wn

# see: http://brandonrose.org/clustering#K-means-clustering

def tryWord(word):
	if not wn.synsets(word):
		return False
	else:
		return True


def createDocVocab(data):
	things = []
	sentences = []
	for doc in twentyNewsTrain.data:
		things.append(doc.split(". "))

	# print(doc)
		
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
twentyNewsTrain = fetch_20newsgroups(subset='train', categories= subcats, shuffle=True, random_state=42, remove=('headers'))

documents = createDocVocab(twentyNewsTrain.data)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 20
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster " + str(i) + ":");
    for ind in order_centroids[i, :10]:
        print(terms[ind])
    print
