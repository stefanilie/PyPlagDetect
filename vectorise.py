from helper import Helper
from collections import Counter
from compiler.ast import flatten
from nltk.probability import FreqDist
from textstat.textstat import textstat

class VectorAnaliser:
    k = 4
    '''
    Constructor
    '''
    def __init__(self, corpus, tagger, stopWords):
        self.corpus = corpus
        self.tagger = tagger
        self.stopWords = stopWords


    def vectorise(self, corpus):
        files= corpus.fileids()

        for file_item in files:
            sentences = corpus.sents(fileids=file_item)
            for index, sentence in enumerate(sentences):
                if index+k/2 <= len(sentences):
                    arr_sentences = sentences[index-k/2:index+k/2]
                    compute_word_frequency(arr_sentences)
                    compute_punctuation(arr_sentences)
                    compute_POS(arr_sentences)
                    compute_pronouns(arr_sentences)
                    if index <= k/2:
