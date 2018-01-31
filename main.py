import os
import re
import math
import nltk
import config
import string
import pprint

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict, treebank
from nltk.tag import UnigramTagger

from paragraph import ParagraphAnalyser
# load the resources


'''
Downloads all of the nltk resources.
'''
def downloadNLTKResources():
    nltk.download('all')


def main():

    # downloadNLTKResources()

    # setting the PATH
    os.chdir(config.PATH)

    # initialising the corpus reader to the docs path
    corpusReader = PlaintextCorpusReader(config.PATH, '.*\.txt')


    # setting stopwords
    stopWords = set(stopwords.words('english'))

    cmdict = cmudict.dict()

    pretty_printer = pprint.PrettyPrinter(indent=4)

    # Training a unigram part of speech tagger
    train_sents = treebank.tagged_sents()[:5000]
    tagger = UnigramTagger(train_sents)

    analyser = ParagraphAnalyser(corpusReader, tagger, stopWords)
    feature_arr = analyser.compute_paragraph_features(corpus=corpusReader)
    feature_arr = analyser.classify_chunks_paragraph(feature_dict=feature_arr,
                                                    corpus=corpusReader)
    print "\n\===============Data after feature classification===============\n"
    pretty_printer.pprint(feature_arr)


    for item in feature_arr:
        if "plagiarized_doc" in item:
            print "\nDocument is plagiarised with a ratio of:"+ \
            str(item["plagiarised_ratio"])

if __name__ == "__main__": main()
