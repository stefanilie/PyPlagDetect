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


from src.paragraph import ParagraphAnalyser
from src.sentence import SentenceAnalyser
from src.vectorise import VectorAnaliser
# load the resources


'''
Downloads all of the nltk resources.
'''
def downloadNLTKResources():
    nltk.download('all')

def exitWithMessage(message): 
    print message
    sys.exit()

def main():
    # downloadNLTKResources()
    mode = ""
    sent_ratios = []
    para_ratios = []

    argv = sys.argv
    if len(argv) != 2 or (argv[1] not in ["para", "sent", "vector", "both"]):
        exitWithMessage("Usage: python main.py [module] - *para*, *sent*, *vector*, *all*")
    else:
        mode = argv[1]

        # setting the PATH
        os.chdir(SUSPICIOUS)

        # initialising the corpus reader to the docs path
        corpusReader = PlaintextCorpusReader(SUSPICIOUS, '.*\.txt')

        # setting stopwords
        stopWords = set(stopwords.words('english'))

        # cmdict = cmudict.dict()

        pretty_printer = pprint.PrettyPrinter(indent=2)

        # Training a unigram part of speech tagger
        # train_sents = treebank.tagged_sents()
        # tagger = UnigramTagger(train_sents)
        # TODO: train this tagger with a huge corpus.

        if mode == "vector":
            isReady = False
            decision = ''

            # basic menu
            while(not isReady):
                print "1. Tokenize, train and export dump via Pickle"
                print "2. Import using Pickle"
                try:
                    decision = raw_input("Choose action: ")
                    decision = int(decision)
                except ValueError:
                    print "Option '{0}' doesn't exist.".format(decision)
                if decision not in [1, 2]:
                    print "Option '{0}' doesn't exist.".format(decision)
                else:
                    isReady = True
            if decision==1:
                taggerReader = PlaintextCorpusReader(OANC, '.*\.txt')        
                vector_analizer = VectorAnaliser(taggerReader, stopWords)
                vector_analizer.vectorise(corpusReader, should_tokenize_corpuses=True)
            elif decision==2:
                vector_analizer = VectorAnaliser(corpusReader, stopWords)
                vector_analizer.vectorise(corpusReader)
            else:
                print "Option '{0}' doesn't exist.".format(decision)


        # Analizing paragraphs for features and outputting an object.
        if mode == "para" or mode == "all":
            para_analyser = ParagraphAnalyser(corpusReader, tagger, stopWords)
            feature_arr = para_analyser.compute_paragraph_features(corpus=corpusReader)
            feature_arr = para_analyser.classify_chunks_paragraph(feature_dict=feature_arr,
                                                        corpus=corpusReader)
            for index, item in enumerate(feature_arr):
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
        # print "\n\===============Data after feature classification===============\n"
        # pretty_printer.pprint(feature_arr)

        if mode == "sent" or mode == "all":
            sentence_analiyser = SentenceAnalyser(corpusReader, tagger, stopWords)
            sent_feat_arr = sentence_analiyser.compute_sentence_features(corpusReader)
            sent_feat_arr = sentence_analiyser.classify_chunks_sentence(sent_feat_arr, corpusReader)

            for index, item in enumerate(sent_feat_arr):
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

        if mode == "para" or mode == "all":
            for index, item in enumerate(para_ratios):
                print "for index ", index , ""
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
