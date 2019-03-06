import os
import re
import sys
import math
import nltk
from src.config import SUSPICIOUS, TRAINING
import string
import pprint

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict, treebank
from nltk.tag import UnigramTagger

from src.paragraph import ParagraphAnalyser
from src.sentence import SentenceAnalyser
from src.vectorise import VectorAnaliser
# load the resources


'''
Downloads all of the nltk resources.
'''
def downloadNLTKResources():
    nltk.download('all')


def main():
    # downloadNLTKResources()
    mode = ""
    sent_ratios = []
    para_ratios = []

    argv = sys.argv
    if len(argv) != 2 or (argv[1] not in ["para", "sent", "vector", "both"]):
        print "Usage: python main.py [module] - *para*, *sent*, *vector*, *all*"
        sys.exit()
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
        train_sents = treebank.tagged_sents()[:5000]
        tagger = UnigramTagger(train_sents)

        # vectorise = VectorAnaliser(corpusReader, tagger, stopWords)
        # vectorise.vectorise(corpus=corpusReader, coeficient=4)

        if mode == "vector":
            decision = ''
            print "1. Tokenize and export dump via Pickle"
            print "2. Import from Pickle"
            decision = int(raw_input("Choose action: "))

            if decision==1:
                trainingCorpusReader=PlaintextCorpusReader(TRAINING, '.*\.txt')
                vector_analizer = VectorAnaliser(trainingCorpusReader, tagger, stopWords)
                vector_analizer.vectorise(corpusReader, should_tokenize_corpuses=True)
            if decision==2:
                vector_analizer = VectorAnaliser(corpusReader, tagger, stopWords)
                vector_analizer.vectorise(corpusReader)


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
