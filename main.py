import os
import re
import sys
import math
import nltk
from src.config import SUSPICIOUS, TRAINING, OANC
import string
import pprint

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict, treebank
from nltk.tag import UnigramTagger

from src.vectorise import VectorAnaliser
# load the resources

def downloadNLTKResources():
    '''
    Downloads all of the nltk resources.
    '''
    nltk.download('all')

def exitWithMessage(message): 
    print message
    sys.exit()

def main():
    # downloadNLTKResources()

    # setting the PATH
    os.chdir(SUSPICIOUS)

    # initialising the corpus reader to the docs path
    corpusReader = PlaintextCorpusReader(SUSPICIOUS, '.*\.txt')

    # setting stopwords
    stopWords = set(stopwords.words('english'))

    # cmdict = cmudict.dict()

    pretty_printer = pprint.PrettyPrinter(indent=2)

    # Training a unigram part of speech tagger
    # TODO: train this tagger with a huge corpus.
    # train_sents = treebank.tagged_sents()
    # tagger = UnigramTagger(train_sents)

    isReady = False
    decision = ''

    # basic menu
    print "Welcome to PyPlagDetect!"
    print "Please choose an action:"
    while(not isReady):
        print "1. Tokenize and export dump via Pickle"
        print "2. Import dumps using Pickle and Analyze PAN corpus"
        print "3. Analyse files (without precision output)"
        try:
            decision = raw_input("Choose action: ")
            decision = int(decision)
        except ValueError:
            print "Option '{0}' doesn't exist.".format(decision)
        if decision not in [1, 2, 3]:
            print "Option '{0}' doesn't exist.".format(decision)
        else:
            isReady = True

    if decision==1:
        trainingCorpusReader=PlaintextCorpusReader(TRAINING, '.*\.txt')
        vector_analizer = VectorAnaliser(trainingCorpusReader, stopWords)
        vector_analizer.should_tokenize(should_tokenize_corpuses=True)
    elif decision==2:
        vector_analizer = VectorAnaliser(corpusReader, stopWords)
        vector_analizer.vectorise(corpusReader)
    elif decision==3:
        vector_analizer = VectorAnaliser(corpusReader, stopWords, custom_mode=True)
        vector_analizer.vectorise(corpusReader)
    else:
        print "Option '{0}' doesn't exist.".format(decision)

if __name__ == "__main__": main()
