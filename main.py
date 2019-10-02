import os
import re
import sys
import math
import nltk
import string

from nltk.corpus import stopwords
from nltk.tag import UnigramTagger
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict, treebank
from nltk.corpus.reader import PlaintextCorpusReader
from src.config import SUSPICIOUS, TRAINING, OANC, SUSPICIOUS_DOCUMENTS, CUSTOM_FOLDER

from src.vectorise import VectorAnaliser
# load the resources

def downloadNLTKResources():
    """
    Downloads all of the nltk resources.
    """
    nltk.download('all')

def exitWithMessage(message): 
    print(message)
    sys.exit()

def main():
    # downloadNLTKResources()

    # setting stopwords
    stopWords = set(stopwords.words('english'))

    isReady = False
    decision = ''

    # basic menu
    print("Welcome to PyPlagDetect!")
    print("Please choose an action:")
    while(not isReady):
        print("1. Tokenize and export dump via Pickle")
        print("2. Import dumps using Pickle and Analyze PAN corpus")
        print("3. Analyse files (without precision output)")
        try:
            decision = input("Choose action: ")
            decision = int(decision)
        except ValueError:
            print("Option '{0}' doesn't exist.".format(decision))
        if decision not in [1, 2, 3]:
            print("Option '{0}' doesn't exist.".format(decision))
        else:
            isReady = True

    if decision==1:
        # TODO: first check if /wiki contains the wiki dump file
        trainingCorpusReader=PlaintextCorpusReader(OANC, '.*\.txt')
        vector_analizer = VectorAnaliser(trainingCorpusReader, stopWords)
        vector_analizer.should_tokenize(should_tokenize_corpuses=True)
    elif decision==2:
        isReady = False
        decision = ''
        print("\nPlease choose folder to analize:")
        while(not isReady):
            print("1. /suspicious: (21 files, 2 without real plag data)")
            print("2. /suspicious-documents: PAN 2009 corpus")
            try:
                decision = input("Choose mode: ")
                decision = int(decision)
            except ValueError:
                print("Option '{0}' doesn't exist.".format(decision))
            if decision not in [1, 2, 3]:
                print("Option '{0}' doesn't exist.".format(decision))
            else:
                isReady = True
        isReady = False
        multi = ''
        print("\nPlease choose a mode:")
        while(not isReady):
            print("1. Single thread")
            print("2. Multi-threading")
            try:
                multi = input("Choose mode: ")
                multi = int(multi)
            except ValueError:
                print("Option '{0}' doesn't exist.".format(multi))
            if multi not in [1, 2, 3]:
                print("Option '{0}' doesn't exist.".format(multi))
            else:
                isReady = True

        multiprocessing = True if multi == 2 else False
        if decision == 1:
            os.chdir(SUSPICIOUS)
            corpusReader = PlaintextCorpusReader(SUSPICIOUS, '.*\.txt')
            
            vector_analizer = VectorAnaliser(corpusReader, stopWords)
            vector_analizer.vectorise(corpusReader, multiprocessing=multiprocessing)
        elif decision == 2:
            os.chdir(SUSPICIOUS_DOCUMENTS)
            corpusReader = PlaintextCorpusReader(SUSPICIOUS_DOCUMENTS, '.*\.txt')
                
            vector_analizer = VectorAnaliser(corpusReader, stopWords)
            vector_analizer.vectorise(corpusReader, multiprocessing=multiprocessing)
    elif decision==3:
        corpusReader = PlaintextCorpusReader(CUSTOM_FOLDER, '.*\.txt')
        vector_analizer = VectorAnaliser(corpusReader, stopWords, custom_mode=True)
        vector_analizer.vectorise(corpusReader)
    else:
        print("Option '{0}' doesn't exist.".format(decision))

if __name__ == "__main__": main()
