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


    def sliding_window(self, sequence, winSize, step=1):
        """Returns a generator that will iterate through
        the defined chunks of input sequence.  Input sequence
        must be iterable."""
    
        # Verify the inputs
        try: it = iter(sequence)
        except TypeError:
            raise Exception("**ERROR** sequence must be iterable.")
        if not ((type(winSize) == type(0)) and (type(step) == type(0))):
            raise Exception("**ERROR** type(winSize) and type(step) must be int.")
        if step > winSize:
            raise Exception("**ERROR** step must not be larger than winSize.")
        if winSize > len(sequence):
            raise Exception("**ERROR** winSize must not be larger than sequence length.")
    
        # Pre-compute number of chunks to emit
        numOfChunks = ((len(sequence)-winSize)/step)+1
    
        # Do the work
        for i in range(0,numOfChunks*step,step):
            yield sequence[i:i+winSize]


    '''
    Calculates average word frequency class.
    @param words - [array string] words that need to be analysed.
    '''
    def average_word_frequecy_class(self, words, most_common_word_freq, suspicious_freq_dist):
        awf = []
        pcf = []
        window_freq_dist = FreqDist(words)
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
        suspicious_freq_dist = FreqDist(self.suspect_corpus_tokenized)
        most_common_word_freq = FreqDist(self.tokenized).most_common(1)[0][1]

        # temporary value for k.
        # will be changed after developing a learning algorithm.
        k=coeficient
        for file_item in files:
            # TODO: replace with nltk.sent_tokenizer
            sentences = corpus.sents(fileids=file_item)
            windows = self.slidingWindow(sentences, k)
            for window in windows:
                self.average_word_frequecy_class(flatten(window), most_common_word_freq, suspicious_freq_dist)

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
