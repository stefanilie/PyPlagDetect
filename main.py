import os
import nltk
import config

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict, treebank
from nltk.tag import UnigramTagger

# load the resources
def downloadNLTKResources():
    nltk.download('stopwords')
    nltk.download('cmudict')
    nltk.download('treebank')

# setting the PATH
os.chdir(config.PATH)

# initialising the corpus reader to the docs path
corpusReader = PlaintextCorpusReader(config.PATH, '.*\.txt')
downloadNLTKResources()

# setting stopwords
stopWords = set(stopwords.words('english'))

cmdict = cmudict.dict()
# print(cmudict.entries()[653:659])

# Training a unigram part of speech tagger 
train_sents = treebank.tagged_sents()[:5000]
tagger = UnigramTagger(train_sents)
print(tagger.tag(treebank.sents()[0]))
