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
from sentence import SentenceAnalyser
# load the resources


'''
Downloads all of the nltk resources.
'''
def downloadNLTKResources():
    nltk.download('all')


def main():

    downloadNLTKResources()

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

    para_analyser = ParagraphAnalyser(corpusReader, tagger, stopWords)
    feature_arr = para_analyser.compute_paragraph_features(corpus=corpusReader)
    feature_arr = para_analyser.classify_chunks_paragraph(feature_dict=feature_arr,
                                                    corpus=corpusReader)
    # print "\n\===============Data after feature classification===============\n"
    # pretty_printer.pprint(feature_arr)

    sentence_analiyser = SentenceAnalyser(corpusReader, tagger, stopWords)
    sent_feat_arr = sentence_analiyser.compute_sentence_features(corpusReader)
    sent_feat_arr = sentence_analiyser.classify_chunks_sentence(sent_feat_arr, corpusReader)

    sent_ratios = []
    para_ratios = []

    for index, item in enumerate(feature_arr):
        if "plagiarized_doc" in item:
            sent_ratios.append({
                "document_no": index,
                "ratio": item["plagiarised_ratio"]
            })
        else:
            sent_ratios.append({
                "document_no": index,
                "ratio": 0

            })
    for index, item in enumerate(sent_feat_arr):
        if "plagiarized_doc" in item:
            para_ratios.append({
                "document_no": index,
                "ratio": item["plagiarised_ratio"]
            })
        else:
            para_ratios.append({
                "document_no": index,
                "ratio": 0
            })
    for index, item in enumerate(para_ratios):
        ratio = str(float(item["ratio"])+float(sent_ratios[index]["ratio"])/2.0)
        if ratio > 0:
            if item["ratio"]>0:
                if sent_ratios[index]["ratio"]>0:
                    print "Document "+str(index) + " is plagiarised with an average ratio of: " + ratio
                else:
                    print "Document "+str(index) + " is plagiarised with an average ratio of (sentence_ratio: 0): " + ratio
            else:
                if sent_ratios[index]["ratio"]>0:
                    print "Document "+str(index) + " is plagiarised with an average ratio of (para_ratio: 0): " + ratio
        else:
            print "Document is not plagiarised."





if __name__ == "__main__": main()
