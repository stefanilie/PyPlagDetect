from helper import Helper
from collections import Counter
from compiler.ast import flatten
from nltk.probability import FreqDist
from textstat.textstat import textstat
from nltk.corpus import movie_reviews, abc, brown, gutenberg, reuters, inaugural

class VectorAnaliser:
    k = 4
    '''
    Constructor
    '''
    def __init__(self, corpus, tagger, stopWords):
        self.corpus = corpus
        self.tagger = tagger
        self.stopWords = stopWords
        self.tokenized = []

    '''
    Tokenizes all corpuses and generates a Frequency Distribution.
    @return [nltk.FreqDest] Frequency Distributiion
    ToDo: add pickle for storage after execution.
    '''
    def tokenize_corpuses(self):
        self.tokenized += Helper.tokenize_corpus(gutenberg, self.stopWords)
        self.tokenized += Helper.tokenize_corpus(movie_reviews, self.stopWords)
        self.tokenized += Helper.tokenize_corpus(abc, self.stopWords)
        self.tokenized += Helper.tokenize_corpus(brown, self.stopWords, True)
        self.tokenized += Helper.tokenize_corpus(reuters, self.stopWords, True)
        print FreqDist(self.tokenized).most_common(10)



    '''
    Main method for vectorising the corpus.
    @param corpus:
    '''
    def vectorise(self, corpus, coeficient):
        files= corpus.fileids()
        # temporary value for k.
        # will be changed after developing a learning algorithm.
        k=coeficient
        for file_item in files:
            sentences = corpus.sents(fileids=file_item)
            for index, sentence in enumerate(sentences):
                arr_sentences = []
                '''
                Window is represented by all k+1 items.
                Until index is equal to k/2, it will jump.
                After, it will pass through until index+k is equal to length.
                '''
                if index-k/2 >=0 and index+k/2<=len(sentences):
                    arr_sentences = sentences[index-k/2:index+k/2]


    #             compute_word_frequency(arr_sentences)
    #             compute_punctuation(arr_sentences)
    #             compute_POS(arr_sentences)
    #             compute_pronouns(arr_sentences)
    #             compute_closed_class_words(arr_sentences)
    #
    # '''
    # Computes the number of punctuation signs
    # '''
    # def compute_punctuation(self, arr_sentences):
    #     return False
