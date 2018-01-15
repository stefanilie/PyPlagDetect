import os
import nltk
import config

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict, treebank
from nltk.tag import UnigramTagger
from nltk.probability import FreqDist

# load the resources
def downloadNLTKResources():
    nltk.download('all')

def getPercentage(words, percentage):
    int(percentage*len(words))

def computeParagraphFeatures(corpus):
     files = corpus.fileids()
     for item in files:
         words = corpus.words(fileids=item)
         # sents = corpus.sents(fileids=item)
         fdist = FreqDist(words)
         percentage = getPercentage(words=words, percentage=0.24)
         print len(fdist.most_common(percentage))

def computeTF_IDF(term, document):
    tf = computeTF(term, document)
    idf = computeIDF(term, corpus)
    return tf*idf

# def computeTF(term, document):


# setting the PATH
os.chdir(config.PATH)

# initialising the corpus reader to the docs path
corpusReader = PlaintextCorpusReader(config.PATH, '.*\.txt')
# downloadNLTKResources()

# setting stopwords
stopWords = set(stopwords.words('english'))

cmdict = cmudict.dict()
# print(cmudict.entries()[653:659])

# Training a unigram part of speech tagger
train_sents = treebank.tagged_sents()[:5000]
tagger = UnigramTagger(train_sents)
computeParagraphFeatures(corpus=corpusReader)


# term-frequency of a word (tf) = 1 + log(frequency of word in a document)
# inverse-document-frequency of a word (idf) = log(Total Number of Documents in the Corpus / Number of Documents in which the word appears)
# tf-idf = tf * idf
