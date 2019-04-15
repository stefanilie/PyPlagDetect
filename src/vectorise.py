import pdb
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
    Calculates average word and punctuation classes.
    @param words - [array string] words that need to be analysed.
    @param most_common_word_freq - [int] frequency of the most common word in a huge corpus
    @param suspicious_freq_dist - [array FreqDist] freq distribution of the suspect corpus

    TODO: check if dividing by word_freq is right. Might need to see value in relation 
    to huge corpus not window. 
    '''
    def average_word_frequecy_class(self, words, most_common_word_freq, suspicious_freq_dist):
        toReturn=[]
        awf = []
        pcf = []
        window_freq_dist = FreqDist(words)
        for word in words:
            if word in string.punctuation:
                pcf.append(window_freq_dist[word])   
            else:
                word_freq = 1 if not suspicious_freq_dist[word] else suspicious_freq_dist[word]
                awf.append(log(float(most_common_word_freq)/word_freq)/log(2))         

        toReturn.append(Helper.normalize_vector(awf))
        toReturn.append(Helper.normalize_vector(pcf))


        return toReturn
        
    '''
    Return FreqDist of all POS tokenized sentences.
    TODO: find a way to normalize this vector to unitary value
    '''
    def compute_POS(self, sentences):
        toReturn=[]
        arr_tagged_sents=[]
        pronouns = []
        arr_stop_words = []
        words_freq_dist = FreqDist(flatten(sentences))
        for sentence in sentences:
            tagged_sent = self.tagger.tag(sentence)
            for word, tag in tagged_sent:
                if word in self.stop_words:
                    arr_stop_words.append(words_freq_dist[word])
                if tag == 'PRP':
                    pronouns.append(words_freq_dist[word])
            arr_tagged_sents.extend(tagged_sent)

        toReturn.append(Helper.normalize_vector(FreqDist(flatten(arr_tagged_sents)).values()))
        toReturn.append(Helper.normalize_vector(pronouns))
        toReturn.append(Helper.normalize_vector(arr_stop_words))

        # TODO: remove None posibility for the below FreqDist
        # TODO: see why we don;t detect pronouns and see how can we fix this.
        return toReturn

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

        # Tokenizing suspicious corpus and getting most common from HUGE corpus.
        files = corpus.fileids()
        self.suspect_corpus_tokenized = Helper.tokenize_corpus(corpus, self.stop_words, with_stop_words=True)
        suspicious_freq_dist = FreqDist(self.suspect_corpus_tokenized)
        most_common_word_freq = FreqDist(self.tokenized).most_common(1)[0][1]

        # temporary value for k.
        # will be changed after developing a learning algorithm.
        k=coeficient
        for file_item in files:
            windows_total = []
            doc_mean_vector = []

            # TODO: replace with nltk.sent_tokenizer
            sentences = corpus.sents(fileids=file_item)
            windows = self.sliding_window(sentences, k)
            # for window in windows:
            #     windows_total.append(self.average_word_frequecy_class(flatten(window), most_common_word_freq, suspicious_freq_dist))
            #     windows_total[-1].extend(self.compute_POS(window))
            #     windows_total[-1] = Helper.normalize_vector(windows_total[-1])
            
            # doc_mean_vector.append(Helper.normalize_vector(windows_total))
            # dict_cosine_similarity = Helper.compute_cosine_similarity_array(windows_total, doc_mean_vector)
            
            for window in windows:
                windows_total.append(self.average_word_frequecy_class(flatten(window), most_common_word_freq, suspicious_freq_dist))
                windows_total[-1].extend(self.compute_POS(window))
                doc_mean_vector.append(Helper.normalize_vector(windows_total[-1]))
            dict_cosine_similarity = Helper.compute_cosine_similarity_array(windows_total, doc_mean_vector)
            


