# -*- coding: utf-8 -*-
import pdb
import os
import sys
import pickle
import numpy as np
from math import e, log
from itertools import groupby
from numpy.linalg import norm
from nltk import word_tokenize
from operator import itemgetter
from compiler.ast import flatten
from nltk.probability import FreqDist
from src.config import SUSPICIOUS, DUMPS
from sklearn.preprocessing import normalize
from nltk.corpus.util import LazyCorpusLoader
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus.reader.tagged import CategorizedTaggedCorpusReader

class Helper:

    '''
    Constructor
    '''
    def __init__(self, corpus, tagger, stopWords):
        self.corpus = corpus
        self.tagger = tagger
        self.stopWords = stopWords

    '''
    Returns the exact number of words based on the percentage needed.
    @param words - [array of strings] words in analysed structure.
    @param percentage - [double] fraction of resource that needs to be extracted.
    @return [int] exact number of words that have to be taken into account.
    '''
    @staticmethod
    def get_percentage(words, percentage):
        return int(percentage * len(words))


    '''
    Intersection of two provided lists.
    @param list1 - [list]
    @param list2 - [list]
    @return [list] intersection of the two lists.
    '''
    @staticmethod
    def get_intersection(list1, list2):
        list1 = flatten(list1)
        list2 = flatten(list2)

        return list(set(list1).intersection(set(list2)))

    '''
    Difference of two provided lists.
    @param list1 - [list]
    @param list2 - [list]
    @return [list] difference of the two lists.
    '''
    @staticmethod
    def get_difference(list1, list2):
        list1 = flatten(list1)
        list2 = flatten(list2)

        return list(set(list1).difference(set(list2)))

    '''
    Maps each POS to a number so that we can analyze the text.
    '''
    @staticmethod
    def switch_pos(x):
        return {
            'NN': 1.0, #noun
            'NNS': 1.5, #noun plural
            'NNP': 2.0, # proper noun
            'NPS': 2.5, #proper noun plural
            'PRP': 3.0, #personal pronoun
            'PP$': 3.5, #possesive pronoun
            'RB': 4.0, #adverb
            'RBR': 4.3, #adverb comparative
            'RBS': 4.9, #adverb superlative
            'VB': 5.0, #verb be, base form
            'VBD': 5.2, #verb be, past tense
            'VBG': 5.4, #verb be gerund
            'VBN': 5.6, #verb be past participle
            'VBP': 5.8, #verb be present 3rd
            'VBZ': 5.99, #verb be 3rd sing. pres.
            'VH': 6.0, #verb have, base form 
            'VHD': 6.2, #verb have, past
            'VHG': 6.4, #verb have, gerund
            'VHN': 6.6, #verb have, past participle
            'VHP': 6.8, #verb have, sing, pressend, non3rd
            'VHZ': 6.99, #verb have, 3rd pers. sing, presens
            'VV': 7.0, #verb base form
            'VVD': 7.2, #verb past tense
            'VVG': 7.4, #verb gerund/past participle
            'VVN': 7.6, #verb past participle
            'VVP': 7.8, #verb sing. present non3rd
            'VVZ': 7.99, #verb 3rd person sing
            'CC': 8.0, #coordinating conjunction
            'CD': 9.0, #cardinal number
            'DT': 10.0, #determiner
            'IN': 11.0, #preposition
            'WDT': 12.0, #wh-determiner
            'WP': 12.2, #pwh pronoun
            'WP$': 12.4, #possesive wh-pronoun
            'WRB': 12.6, #wh-adverb
            'JJ': 13.0, #adjective
            'JJR': 13.4, #adjective, comparative
            'JJS': 13.8, #adjective, superlative
            'MD': 14.0, #modal
            'POS': 15.0, #possesive ending
            # 'SENT': 16.0, #sencencebreak (pct)
            # 'SYM': 16.0, #symbols
            'TO': 17.0
        }.get(x, 0)

    '''
    Method obtained from
    https://github.com/ypeels/nltk-book/blob/master/exercises/2.21-syllable-count.py
    Calculates syllable count for the provided word.
    @param word - string representing the word.
    \=======================DEPRECATED========================/
    '''
    @staticmethod
    def syllables_in_word(word):
        flat_dict = dict(cmudict.entries())
        if flat_dict.has_key(word):
            return len([ph for ph in flat_dict[word] if ph.strip(string.letters)])
        else:
            return 0


    '''
    First it calculates the percentage of a feature relative to the bigger one.
    After this it edits the name of the feature so that it contains
    the "_percentage" component.
    @param relative_to - [int] the feature to which we calculate the percentage of
    the other smaller ones. Example: paragraph_words_count.
    @param feature - [dict] feature dictionary containing the values to be calculated.
    '''
    @staticmethod
    def get_feature_percentage(relative_to, feature):
        dict_reply = {}
        for key, value in feature.iteritems():
            dict_reply.update({str(key)+"_percentage": np.true_divide(100*value, relative_to)})
        return dict_reply

    '''
    Computes the Term Frequency (TF).
    @param term - [string] the term who's TF we're computing.
    @param tokenized_document - [list string] can be either the sentence,
    the paragraph, or even the entire document. Based on this we calculate the
    TF for the according instance.
    @return [int] value of the TF.
    '''
    @staticmethod
    def compute_TF(term, tokenized_document):
        return 1 + log(tokenized_document.count(term))

    @staticmethod
    def get_overlap(a, b):
      '''
      Computes the overlap of the provided intervals.
      Returns number of shared items.
      '''
      return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    '''
    Computes the Inverse Term Frequency (IDF) coeficient.
    IDF = log(Nr of Docs in the Corpus / Nr of Docs in which the word appears).
    @param term - [string] term to calculate the idf for.
    @param tokenized_document - [list of list string] it can be document.
    @return [int] value of the IDF.
    '''
    @staticmethod
    def compute_IDF(term, tokenized_document):
        doc_number=0
        # Iterating the paragraphs.
        for doc in tokenized_document:
            if term in doc:
                doc_number += 1
        return log(len(np.true_divide(tokenized_document, doc_number)))


    '''
    Computes the TF-IDF value.
    @param term - [string] the term to calculate the tf-idf value.
    @param document - [list of string] document or array of docs that needs to be
    calculated.
    @return [int] - value of the computed Tf-Idf
    '''
    @staticmethod
    def compute_TF_IDF(term, document):
        tf = Helper.compute_TF(term, document)
        idf = Helper.compute_IDF(term, document)
        return tf * idf

    '''
    Tokenizes all files from a corpus.
    @param corpus - [nltk.corpus]
    @param stopWords - [nltk.stopWords] array containing all eng stopwords.
    @param cathegorized - [Boolean] if corpus is categorized or not.
    @param with_stop_words - [Boolean] if stop words should be excluded or not.
    @return [list of strings] - tokenized array for all the corpus docs.
    '''
    @staticmethod
    def tokenize_corpus(corpus, stopWords, cathegorized=False, with_stop_words=False):
        print "==========="
        print "Tokenizeing ", corpus
        tokenized = []
        if not cathegorized: 
            for id in corpus.fileids():
                print "1-------file------"
                print id
                raw = corpus.raw(id)
                tokenized += word_tokenize(raw)        
        else:
            print "2-------cathegory------"
            print type(corpus)
            if type(corpus) is LazyCorpusLoader:
                tokenized += corpus.words()
            else:
                tokenized += word_tokenize(corpus.raw())       
        if not with_stop_words:
            tokenized = Helper.get_difference(tokenized, stopWords)
        return tokenized

    '''
    Returns FreqDist of the tokenized suspect file.
    '''
    @staticmethod
    def tokenize_file(corpus, stopWords, fileId, with_stop_words=False):
        raw = corpus.raw(fileId)
        tokenized = word_tokenize(raw)
        if not with_stop_words:
            tokenized = Helper.get_difference(tokenized, stopWords)
        return FreqDist(tokenized)

    '''
    Creates data dump for tokenization to destination file.
    @param tokenized - [list of strings] Tokenized array of words.
    @param destination - [string] File on which the data will be written.
    '''
    @staticmethod
    def create_dump(tokenized, destination):
        # saving current directory
        current_directory=os.getcwd()
        
        # chaning it to the data dumps one
        os.chdir(DUMPS)

        # dumping data with pickle
        save_tokenized=open(destination, "wb")
        pickle.dump(tokenized, save_tokenized)
        save_tokenized.close()

        # reverting to previous directory
        os.chdir(current_directory)

    '''
    Reads data dump of tokenized corpus/
    @param file_name - [string] File name of the data dump.
    @returns tokenized_dump - [string array] Tokenized words. 
    '''
    @staticmethod
    def read_dump(file_name):
         # saving current directory
        current_directory=os.getcwd()
        
        # chaning it to the data dumps one
        os.chdir(DUMPS)

        tokenized_file = open(file_name, "rb")
        tokenized_dump = pickle.load(tokenized_file)
        tokenized_file.close()

        os.chdir(SUSPICIOUS)

        return tokenized_dump

    '''
    Finds all consecutive items in array.
    Returns grouped consecutive items.
    '''
    @staticmethod
    def find_consecutive_numbers(arr):
        toReturn = []
        for k, g in groupby(enumerate(arr), lambda (i, x): i-x):
            toReturn.append(map(itemgetter(1), g))
        return toReturn


    @staticmethod
    def normalize_vector(vector):
        # pdb.set_trace()
        # return np.linalg.norm(vector)
        return normalize(vector)[0].tolist()

    '''
    Computes standard deviation for the provided sentences array.
    @param sent_array - [array] sentence statistics
    @param cosine_similarity_array - [array] calculated cosine similarity array for all the sentences
    @param mean - [float] mean cosine similarity
    @return [float] standard deviation of the for the annalized widow.
    '''
    @staticmethod
    def stddev(sent_array, cosine_similarity_array, mean):
        sum=0
        print "\nComputing sttdev"
        for index, sent in enumerate(sent_array):
            # TODO: mean and the result of cosine simularity MUST be np.array type (matrices)
            # TODO: check to see .sum methid f  rom numpy
            Helper.print_progress(index, len(sent_array))
    
            sum += np.square(np.array(cosine_similarity_array[index]) - np.array(mean))
        return np.sqrt(np.true_divide(1, len(sent_array))*sum)
            
    '''
    Decides if annalized part of the document is plagiarised or not.
    @param cosine_similarity_value - [float] 
    @param mean - [float]
    @param stddev - [float]
    @return [boolean]
    '''
    @staticmethod
    def trigger_suspect(cosine_similarity_value, mean, stddev):
        return cosine_similarity_value < mean - e*stddev
        
    @staticmethod
    def precision(arr_overlap, arr_plag_offset):
        '''
        true positive/actual results
        '''
        s=0
        if len(arr_plag_offset) == 0:
            return 0
        for index, plag_interval in enumerate(arr_plag_offset):
            plagiarized_chars = plag_interval[1]-plag_interval[0]
            s += np.true_divide(arr_overlap[index], plagiarized_chars)
        return np.true_divide(s, len(arr_plag_offset))

    @staticmethod
    def recall(arr_suspect_overlap, arr_suspect_offset):
        '''
        true positive / predicted results
        '''
        s=0
        # check here if sus.offset has same length ass sus.overlap 
        if len(arr_suspect_offset) == 0:
            return 0
        for index, suspect_interval in enumerate(arr_suspect_offset):
            suspect_chars = suspect_interval[1]-suspect_interval[0]
            s += np.true_divide(arr_suspect_overlap[index], suspect_chars)
        return np.true_divide(s, len(arr_suspect_offset))

    # @staticmethod
    # def accuracy(arr_overlap, )

    @staticmethod
    def granularity_f1(precision, recall, arr_overlap):
        if precision and recall :
            f1 = np.true_divide(2*precision*recall, precision+recall)
            return f1
        else: 
            return 0
    
    @staticmethod
    def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        filled_length = int(round(bar_length * iteration / float(total)))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()

    @staticmethod
    def compute_flesch_reading_ease(words, syllables, sentences):
        '''
        Computes the Flesch reading ease.
        '''
        return 206.835-1.015 * (np.true_divide(words, sentences)) - 84.6 * (np.true_divide(syllables, words))