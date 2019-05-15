import pdb
import pickle
import string
import numpy as np

from math import log
from numpy import dot
from nltk import pos_tag
from helper import Helper
from numpy.linalg import norm
from collections import Counter
from compiler.ast import flatten
from nltk.probability import FreqDist
from textstat.textstat import textstat
from src.results_analyzer import ResultsAnalyzer
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
        self.arr_cosine_similarity = {}
        self.mean = 0.0
        self.standard_deviation = 0.0

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

    -------------------------------------DEPRECATED------------------------------------------
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

    def new_avf(self, sentences, most_common_word_freq, suspicious_freq_dist):
        awf=[]
        pcf=[]
        stp = []
        pos = []
        prn = []

        words = flatten(sentences)
        pos_tagged_sentences = self.compute_POS(sentences)
        fdist = FreqDist(words)
        for item in suspicious_freq_dist:
            isInWindow = True if item in words else False
            if isInWindow: 
                # pos.append tagged value, else 0
                # TODO: see why we needed the below comented line
                word_freq = suspicious_freq_dist[item]
                # word_freq = 1 if not suspicious_freq_dist[item] else suspicious_freq_dist[item]

                awf.append(log(float(most_common_word_freq)/word_freq) / log(2) )         
                pcf.append(fdist[item] if item in string.punctuation else 0)
                stp.append(fdist[item] if item in self.stop_words else 0)
                pos.append(pos_tagged_sentences[item])
                if pos_tagged_sentences[item] == 3.0 or pos_tagged_sentences[item] == 3.5:
                    prn.append(1)
                else:
                    prn.append(0)
                # TODO: check if iterating sentences we have issues with 
                # double values due to iterating also though docFreqDist
                # prn.append(fdist[item] if tag == 'PRP' else 0)
            else:
                awf.append(0)
                pcf.append(0)
                stp.append(0)
                pos.append(0)
                prn.append(0)
        
        awf = Helper.normalize_vector([awf])
        pcf = Helper.normalize_vector([pcf])
        stp = Helper.normalize_vector([stp])
        pos = Helper.normalize_vector([pos])
        prn = Helper.normalize_vector([prn])

        
        # Old way of doing things, it adds vectors to an array        
        # toReturn = np.concatenate((awf, pcf))
        # toReturn = np.concatenate((toReturn, stp))

        toReturn = awf
        toReturn.extend(pcf)
        toReturn.extend(stp)
        toReturn.extend(pos)
        toReturn.extend(prn)
        # return np.array(Helper.normalize_vector([toReturn]))
        return Helper.normalize_vector([toReturn])


    '''
    Return Dictionary with pos tokenized sentences values.
    '''
    def compute_POS(self, sentences):
        dict_pos={}
        for sentence in sentences:
            # TODO: change to use this instead of default pos_tag
            # tagged_sent = self.tagger.tag(sentence)
            tagged_sents = pos_tag(sentence)
            tagged_sents = self.create_pos_dict(tagged_sents)
            
            # write a switch for pronouns also that returns 1 for each occurance.
            dict_pos.update(tagged_sents)

        return dict_pos

    '''
    Iterates sentence and passes through filter each POS.
    Returns dictionary containing {word: pos}
    '''
    def create_pos_dict(self, tagged_sent):
        dict_pos={}
        for (word, pos) in tagged_sent:
            dict_pos[word] = Helper.switch_pos(pos)
        return dict_pos


    '''
    Calculates the cosine similarity between the window sentences and mean document
    feature arrays. Uses scikit learn method for this.
    @param windows - [[array]] statistics for all windiws/sentences in the doc
    @param document - [[array]] statistics for the whole document
    @return dict_reply = [dict] mean cosine similarity and array with all computed cosine similarities 
    '''
    def compute_cosine_similarity_array(self, windows, document):
        cs_sum=0
        cosine_array=[]
        sent_count = len(windows)

        for window in windows:
            cs = dot(window, document)/(norm(window)*norm(document))
            cs_sum+=cs
            cosine_array.append(cs)

        if len(cosine_array):
            mean = np.true_divide(1, sent_count) * cs_sum
            self.mean = mean
            self.arr_cosine_similarity = cosine_array
        

    '''
    Generates suspect passages array by concatenating all consecutive suspect sentences. 
    '''
    def generate_passages(self):
        arr_suspect_index=[]
        for index, cs in enumerate(self.arr_cosine_similarity):
           isSuspect = Helper.trigger_suspect(cs, self.mean, self.standard_deviation)
           if isSuspect:
               arr_suspect_index.append(index)
        
        arr_suspect_index = Helper.find_consecutive_numbers(arr_suspect_index)
        return arr_suspect_index

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

        # # TODO: changing this to point to current file, not whole corpus.
        # suspicious_freq_dist = FreqDist(self.suspect_corpus_tokenized)
        most_common_word_freq = FreqDist(self.tokenized).most_common(1)[0][1]

        # temporary value for k.
        # will be changed after developing a learning algorithm.
        k=coeficient
        for file_item in files:
            windows_total = []
            doc_mean_vector = []

            suspicious_freq_dist = Helper.tokenize_file(corpus, self.stop_words, file_item, True)

            # TODO: replace with nltk.sent_tokenizer
            sentences = corpus.sents(fileids=file_item)
            windows = self.sliding_window(sentences, k)
            
            doc_mean_vector = self.new_avf(sentences, most_common_word_freq, suspicious_freq_dist)
            doc_mean_vector = Helper.normalize_vector([doc_mean_vector])
            
            for index, sentence in enumerate(sentences):
                '''
                Window is represented by all k+1 items.
                Until index is equal to k/2, it will jump.
                After, it will pass through until index+k is equal to length.
                '''
                arr_sentences = []
                if index<k/2:
                    arr_sentences = sentences[:k]
                elif index+k/2 >= len(sentences):
                    arr_sentences = sentences[len(sentences)-k:len(sentences)]
                else:
                    arr_sentences = sentences[index-k/2:index+k/2]                    

                toAppend = self.new_avf(arr_sentences, most_common_word_freq, suspicious_freq_dist)
                # toAppend.extend(self.compute_POS(arr_sentences, suspicious_freq_dist))
                windows_total.append(Helper.normalize_vector([toAppend]))

                # windows_total.append(self.compute_POS(arr_sentences, suspicious_freq_dist))
                # windows_total.append(self.new_avf(flatten(arr_sentences), most_common_word_freq, suspicious_freq_dist))

            # TODO: old way of generating the window.
            # Still works but can't iterate bearing in mind sentences.
            # for window in windows:
                # windows_total.append(self.new_avf(flatten(arr_sentences), most_common_word_freq, suspicious_freq_dist))
            

            # Deprecated method trying to compute everything based on vector norm.
            # for window in windows:
            #     windows_total.append(self.average_word_frequecy_class(flatten(window), most_common_word_freq, suspicious_freq_dist))
            #     windows_total[-1].extend(self.compute_POS(window))
            #     windows_total[-1] = Helper.normalize_vector(windows_total[-1])
           
            # Another deprecated method, tried iterating by sentences not window.
            # for index, sentence in enumerate(sentences):
            #     pdb.set_trace()
            #     if index-k/2 <= 0:
            #         windows_total.append(self.average_word_frequecy_class(flatten(windows[0]), most_common_word_freq, suspicious_freq_dist))
            #     elif index+k/2>len(sentences):
            #         windows_total.append(self.average_word_frequecy_class(flatten(windows[len(windows)-1]), most_common_word_freq, suspicious_freq_dist))
            #     else:
            #         windows_total.append(self.average_word_frequecy_class(flatten(windows[index]), most_common_word_freq, suspicious_freq_dist))            
            
            self.compute_cosine_similarity_array(windows_total, doc_mean_vector)
            # self.mean = self.mean
            # self.arr_cosine_similarity = self.arr_cosine_similarity
            self.standard_deviation = Helper.stddev(windows_total, self.arr_cosine_similarity, self.mean)
            suspect_sentences = []
            dict_suspect_char_count = {}
            for index, cs in enumerate(self.arr_cosine_similarity):
                isSuspect = Helper.trigger_suspect(cs, self.mean, self.standard_deviation)
                suspect_sentences.append(index) if isSuspect else False
                # compute number of chars in one suspect sentece
                dict_suspect_char_count.update({index: sum(map(len, sentences[index]))})
            
            arr_suspect_chunks = Helper.find_consecutive_numbers(suspect_sentences)

            result_analizer = ResultsAnalyzer()
            xml_data = result_analizer.get_offset_from_xml(file_item)
            result_analizer.get_plagiarised(file_item, xml_data["offset"], xml_data["length"])
            # TODO: see if it's fake positive or not

            # Helper.precision(arr_suspect_chunks, dict_suspect_char_count)
            pdb.set_trace()


