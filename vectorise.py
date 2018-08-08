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


    '''
    Main method for vectorising the corpus.
    @param corpus:
    '''
    def vectorise(self, corpus, coeficient):
        files= corpus.fileids()

        for file_item in files:
            sentences = corpus.sents(fileids=file_item)
            for index, sentence in enumerate(sentences):
                arr_sentences = []
                '''
                While the index is < k/2, the window will be until k/2.
                After, it will be the exact size of k.
                '''
                if index-k/2 >= 0:
                    if index+k/2 <= len(sentences)-1:
                        arr_sentences = sentences[index-k/2:index+k/2]
                elif index-k/2 < 0:
                    arr_sentences = sentences[:index+k/2]
                compute_word_frequency(arr_sentences)
                compute_punctuation(arr_sentences)
                compute_POS(arr_sentences)
                compute_pronouns(arr_sentences)
                compute_closed_class_words(arr_sentences)
