import pickle
import string
from math import log
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
    def __init__(self, corpus, tagger, stop_words):
        self.corpus = corpus
        self.tagger = tagger
        self.stop_words = stop_words
        self.tokenized = []
        self.suspect_corpus_tokenized = None

    '''
    Tokenizes all corpuses and generates a Frequency Distribution.
    @return [nltk.FreqDest] Frequency Distributiion
    ToDo: add pickle for storage after execution.
    '''
    def tokenize_corpuses(self, file_name):
        self.tokenized += Helper.tokenize_corpus(gutenberg, self.stop_words)
        self.tokenized += Helper.tokenize_corpus(movie_reviews, self.stop_words)
        self.tokenized += Helper.tokenize_corpus(abc, self.stop_words)
        self.tokenized += Helper.tokenize_corpus(brown, self.stop_words, True)
        self.tokenized += Helper.tokenize_corpus(reuters, self.stop_words, True)
        self.tokenized += Helper.tokenize_corpus(self.corpus, self.stop_words)
        Helper.create_dump(self.tokenized, file_name)


    '''
    Calculates average word frequency class.
    @param words - [array string] words that need to be analysed.
    '''
    def average_word_frequecy_class(self, words):
        awf = []
        pcf = []
        window_freq_dist = FreqDist(words)
        most_common_word_freq = FreqDist(self.tokenized).most_common(1)[0][1]
        suspicious_freq_dist = FreqDist(self.suspect_corpus_tokenized)
        for word in words:
            word_freq = 1 if not suspicious_freq_dist[word] else suspicious_freq_dist[word]
            awf.append(log(float(most_common_word_freq)/word_freq)/log(2))

            if word in string.punctuation:
                pcf.append((word, window_freq_dist[word]))

        print "-------------------"
        print "pcf: ", pcf
    '''
    Main method for vectorising the corpus.
    @param corpus:
    '''
    def vectorise(self, corpus, coeficient=4, should_tokenize_corpuses=False):
        file_name = "tokenized.pickle"
       
        # check if tokenized is done.
        if not len(self.tokenized) and not should_tokenize_corpuses:
            self.tokenized = Helper.read_dump(file_name)
            dist_huge_token = FreqDist(self.tokenized)
        elif not len(self.tokenized) and should_tokenize_corpuses:
            self.tokenize_corpuses(file_name)

        files= corpus.fileids()
        self.suspect_corpus_tokenized = Helper.tokenize_corpus(corpus, self.stop_words, with_stop_words=True)
        # temporary value for k.
        # will be changed after developing a learning algorithm.
        k=coeficient
        for file_item in files:
            # TODO: replace with nltk.sent_tokenizer
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
                    words=[]
                    # TODO: rplace with flatten
                    for sent in arr_sentences:
                        words.extend(sent)
                    self.average_word_frequecy_class(words)
                    # calculate/tokenize corpus with FreqDist
                    # class should have as atribute content of 
                        # tokenized huge corpus
                        # tokenized analisis suspect corpus.

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
