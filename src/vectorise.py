import os
import pdb
import sys
import pickle
import pyphen
import string
import numpy as np
import multiprocessing
import scipy.stats as sc

from math import log, floor
from numpy import dot
from nltk import pos_tag
from helper import Helper
from numpy.linalg import norm
from collections import Counter
from compiler.ast import flatten
from nltk.tag import UnigramTagger
from nltk.probability import FreqDist
from sacremoses import MosesDetokenizer
from config import WIKI_DUMP, TAGGER_DUMP
from nltk import sent_tokenize, word_tokenize
from src.results_analyzer import ResultsAnalyzer
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus import movie_reviews, abc, brown, gutenberg, reuters, inaugural

class VectorAnaliser:
    k = 4
    """
    Constructor
    """
    def __init__(self, corpus, tagger, stop_words):
        self.corpus = corpus
        self.tagger = tagger
        self.stop_words = stop_words
        self.tokenized = []
        self.suspect_corpus_tokenized = None
        self.arr_cosine_similarity = {}
        self.mean = 0.0
        self.standard_deviation = 0.0
        self.md = MosesDetokenizer(lang='en')
        self.arr_plag_offset = []
        self.arr_suspect_offset = []
        self.arr_suspect_overlap = []
        self.dic = pyphen.Pyphen(lang='en_GB')
        self.coca_freq_dict = Helper.setup_coca_dictionary()
        self.missed_words = []
        self.arr_mean_precision = []
        self.arr_mean_recall = []
        self.arr_mean_f1 = []

    def tokenize_corpuses(self, file_name):
        """
        Tokenizes all corpuses and generates a Frequency Distribution.
        @return [nltk.FreqDest] Frequency Distributiion
        """
        self.tokenized += Helper.tokenize_corpus(gutenberg, self.stop_words)
        self.tokenized += Helper.tokenize_corpus(movie_reviews, self.stop_words)
        self.tokenized += Helper.tokenize_corpus(abc, self.stop_words)
        self.tokenized += Helper.tokenize_corpus(brown, self.stop_words, True)
        self.tokenized += Helper.tokenize_corpus(reuters, self.stop_words, True)
        self.tokenized += Helper.tokenize_corpus(self.corpus, self.stop_words)
        Helper.create_dump(self.tokenized, file_name)

    def train_unigram_tagger(self, TAGGER_DUMP):
        '''
        Trains an unigram tagger based on OANC.
        Exports data to a dump file in the dumps folder.
        '''
        toTag = []
        files = self.corpus.fileids()
        print "\nExtracting words from files..."
        for index, file_item in enumerate(files):
            Helper.print_progress(index, len(files))
            paras = self.corpus.paras(file_item)
        
            for sentences in paras:
                st = flatten(sentences)
                training_tags = []
                for word in st:
                    if '_' in word:
                        if len(word.split('_')) > 2:
                            continue
                        w, pos = word.split('_')
                        if w != '' and pos != '':
                            training_tags.append((w, pos))
                if len(toTag) > 0:
                    toTag.append(training_tags) 
        print "\Training tagger..."     
        pos_tagger = UnigramTagger(train=toTag, model=('effect', 'NN'), verbose=True)
        print "\nCreating dump in unigram_tagger.pickle..."
        Helper.create_dump(pos_tagger, TAGGER_DUMP)

    def feature_extraction(self, sentences, most_common_word_freq, suspicious_freq_dist, verbose=False):
        """
        Main method for computing features of the text.
        Iterates words in sentences and computes:
        average word frequency, stopwords,
        punctuation, POS tags, and pronouns.

        @return normalized feature vector
        """
        # create empty vectors for all features
        # will have length of fdist of the document
        awf = [0] * len(suspicious_freq_dist) # average word frq
        pcf = [0] * len(suspicious_freq_dist) # punctuation signs
        stp = [0] * len(suspicious_freq_dist) # stopwords
        pos = [0] * len(suspicious_freq_dist) # parts of speech
        prn = [0] * len(suspicious_freq_dist) # pronouns
        awl = [] # average words length
        asl = [0] * len(sentences) # average sentence length
        awps = [0] * len(sentences) # average words per sentence
        hapax = 0 # hapax legomena value
        dis_legomena = 0 # hapax dislegomane value
        awd =  0 # average word diversity (no unique words/no words)
        aspw = [] #average syllable count per word
        fre = 0 # flesch reading ease 
        gre = 0 # flesch kincaid grade level
        sha_entr = [] # shanon entropy value
        toReturn = []
        

        flat_sent = flatten(sentences)
        fdist = FreqDist(flat_sent)
        hapax = np.true_divide(len(fdist.hapaxes()), len(flat_sent))
        awd = np.true_divide(len(fdist), len(flat_sent))

        # computing array for all POS stored in objects
        # each object represents a sentence.
        arr_tagged_pos_per_sent = self.compute_POS(sentences)

        # iterating sentences and then words in them
        for index, words in enumerate(sentences):
            if verbose:
                Helper.print_progress(index, len(sentences))
            for word in words:
                word = word.lower()
                word_freq = suspicious_freq_dist[word]
                word_freq_wiki_corpus = self.tokenized[word]

                if '-' in word and word_freq_wiki_corpus == 0:
                        word_freq_wiki_corpus = np.average([self.tokenized[i] for i in word.split('-')])

                # this coveres the cases where the word isn't part of the 
                # big wiki freq dist.
                if word_freq_wiki_corpus == 0:
                    if word not in self.coca_freq_dict.keys():
                        if word not in self.missed_words:
                            self.missed_words.append(word)
                        word_freq_wiki_corpus = 1
                    else: 
                        word_freq_wiki_corpus = self.coca_freq_dict[word]
                
                if word not in  suspicious_freq_dist.keys():
                    # this is for some weird cases where we have words that
                    # couldn't be tokenized.
                    continue
                item_index = suspicious_freq_dist.keys().index(word)

                if word_freq == 2:
                    dis_legomena += 1

                # computing number of occurances 
                # awf[item_index] = log(float(most_common_word_freq)/word_freq) / log(2)

                # awf[item_index] = floor(log(np.true_divide(most_common_word_freq, word_freq_wiki_corpus))/log(2))
                awf[item_index] = floor(log(np.true_divide(word_freq_wiki_corpus, word_freq))/log(2))
                pcf[item_index] = fdist[word] if word in string.punctuation else 0
                stp[item_index] = fdist[word] if word in self.stop_words else 0
                pos[item_index] = arr_tagged_pos_per_sent[index][word]

                # if it's a pronoun, then we take them into account
                if arr_tagged_pos_per_sent[index][word] == 3.0 or arr_tagged_pos_per_sent[index][word] == 3.5:
                    prn[item_index] = 1
                awl.append(len(word))
                asl[index] += len(word)
                aspw.append(len(self.dic.inserted(word).split('-')))
                sha_entr.append(word_freq*log(word_freq, 2))
            awps[index] = len(words)

        words = np.sum(awps)
        syllables = np.sum(aspw)
        fre, gre = Helper.compute_flesch_reading_ease(words, syllables, len(sentences))
        sha_entr = sc.entropy(sha_entr, None, 2)

        awf = Helper.normalize_vector([awf])
        pcf = Helper.normalize_vector([pcf])
        stp = Helper.normalize_vector([stp])
        pos = Helper.normalize_vector([pos])
        prn = Helper.normalize_vector([prn])

        toReturn.append(np.average(awl))
        toReturn.append(np.average(asl))
        toReturn.append(np.average(awps))
        toReturn.append(np.average(aspw))
        toReturn.append(np.average(awf))
        toReturn.append(hapax)
        toReturn.append(dis_legomena)

        toReturn.append(fre)
        toReturn.append(gre)
        toReturn.append(sha_entr)
        toReturn.append(awd)
        
        toReturn.extend(awf)
        toReturn.extend(pcf)
        toReturn.extend(stp)
        toReturn.extend(pos)
        toReturn.extend(prn)

        return Helper.normalize_vector([toReturn])

    def analize_file(self, file_item, k):
        # Most common word in a big corpus.
        # most_common_word_freq = FreqDist(self.tokenized).most_common(1)[0][1]
        most_common_word_freq = self.tokenized.most_common()[0][1]

        windows_total = []
        doc_mean_vector = []
        dict_all_sentences = {} # used to save all the windows sent for analization.
        dict_offset_index = {} # used for saving the start offset and lenght of each window.
        offset_counter = 0
        self.arr_plag_offset = []
        self.arr_suspect_offset = []
        self.arr_suspect_overlap = []
        self.arr_cosine_similarity = []
        self.mean = 0
        self.standard_deviation = 0

        suspicious_freq_dist = Helper.tokenize_file(self.corpus, self.stop_words, file_item, True)

        # tokenizing the words from the sentences 
        sentences = [word_tokenize(sent) for sent in sent_tokenize(self.corpus.raw(fileids=file_item))]
        
        # computing the document mean vector
        print "\n==========\nanalizing %s" % (file_item)
        print "\nComputing reference_vector"
        doc_mean_vector = self.feature_extraction(sentences, most_common_word_freq, suspicious_freq_dist, True)
        print "\nComputing features"
        for index, sentence in enumerate(sentences):
            """
            Window is represented by all k+1 items.
            Until index is equal to k/2, it will jump.
            After, it will pass through until index+k is equal to length.
            """
            arr_sentences = []
            if index<k/2:
                arr_sentences = sentences[:k]
            elif index+k/2 >= len(sentences):
                arr_sentences = sentences[len(sentences)-k:len(sentences)]
            else:
                arr_sentences = sentences[index-k/2:index+k/2]                    

            Helper.print_progress(index, len(sentences))
            toAppend = self.feature_extraction(arr_sentences, most_common_word_freq, suspicious_freq_dist, False)
            windows_total.append(toAppend)
            
            dict_all_sentences[index] = sentence

            new_offset = len(self.md.detokenize(sentence))
            offset_counter += new_offset
            dict_offset_index[index] = [offset_counter, new_offset]


        # compute cosine similarity for all windows plus mean
        self.compute_cosine_similarity_array(windows_total, doc_mean_vector)  

        # computing the standard deviation 
        self.standard_deviation = Helper.stddev(windows_total, self.arr_cosine_similarity, self.mean)

        arr_suspect_chunks = self.get_suspect_index(sentences)
        self.compare_with_xml(file_item, dict_offset_index, arr_suspect_chunks)

        # if self.arr_plag_offset:
        recall = 0
        precision = 0
        f1 = 0
        if len(self.arr_plag_offset) == 0 and len(self.arr_suspect_offset) == 0:
            # arr_mean_recall.append(1)
            # arr_mean_precision.append(1)
            # arr_mean_f1.append(1)
            # print "\n%s precision: " % (file_item), 1
            # print "%s recall: " % (file_item), 1
            # print "%s f1: " % (file_item), 1
            print "\n%s: No plagiarism detected and none existing" % (file_item)
        else: 
            precision = Helper.precision(self.arr_overlap, self.arr_plag_offset)
            recall = Helper.recall(self.arr_suspect_overlap, self.arr_suspect_offset)
            f1 = Helper.granularity_f1(precision, recall, self.arr_overlap)

            self.arr_mean_recall.append(recall)
            self.arr_mean_precision.append(precision)
            self.arr_mean_f1.append(f1)
        
            print "\n%s precision: " % (file_item), precision
            print "%s recall: " % (file_item), recall
            print "%s f1: " % (file_item), f1

    def compute_POS(self, sentences):
        """
        Return Array with dict per sent of pos tokenized sentences values.
        """
        arr_pos=[]
        for sentence in sentences:
            # TODO: change to use this instead of default pos_tag
            # tagged_sents = self.tagger.tag(sentence)
            # pdb.set_trace()
            tagged_sents = pos_tag(sentence)
            tagged_sents = self.create_pos_dict(tagged_sents)
            
            # append to an array all objects
            arr_pos.append(tagged_sents)

        return arr_pos

    def create_pos_dict(self, tagged_sent):
        """
        Iterates sentence and passes through filter each POS.
        Returns dictionary containing {word: pos}
        """
        dict_pos={}
        for (word, pos) in tagged_sent:
            dict_pos[word.lower()] = Helper.switch_pos(pos)
        return dict_pos

    def compute_cosine_similarity_array(self, windows, document):
        """
        Calculates the cosine similarity between the window sentences and mean document
        feature arrays. Uses scikit learn method for this.
        @param windows - [[array]] statistics for all windiws/sentences in the doc
        @param document - [[array]] statistics for the whole document
        @return dict_reply = [dict] mean cosine similarity and array with all computed cosine similarities 
        """
        cs_sum=0
        cosine_array=[]
        sent_count = len(windows)

        print "\nComputing cosine_similarity"
        for index, window in enumerate(windows):
            Helper.print_progress(index, len(windows))
            cs = dot(window, document)/(norm(window)*norm(document))
            cs_sum+=cs
            cosine_array.append(cs)

        if len(cosine_array):
            mean = np.true_divide(1, sent_count) * cs_sum
            self.mean = mean
            self.arr_cosine_similarity = cosine_array

    def generate_passages(self):
        """
        Generates suspect passages array by concatenating all consecutive suspect sentences. 
        """
        arr_suspect_index=[]
        for index, cs in enumerate(self.arr_cosine_similarity):
           isSuspect = Helper.trigger_suspect(cs, self.mean, self.standard_deviation)
           if isSuspect:
               arr_suspect_index.append(index)
        
        arr_suspect_index = Helper.find_consecutive_numbers(arr_suspect_index)
        return arr_suspect_index

    def get_suspect_index(self, sentences):   
        """
        Checks if the window aligns with the rest of the document.
        Triggers true or false response based on that.
        If is suspect, it adds the index of the sentence to the suspect list.
        @return list of suspect window indexes grouped by consecutive arrays.
        """ 
        suspect_sentences = []
        dict_suspect_char_count = {}
        for index, cs in enumerate(self.arr_cosine_similarity):
            isSuspect = Helper.trigger_suspect(cs, self.mean, self.standard_deviation)
            
            # print "index: %s, is suspect: %s" %(str(index), str(isSuspect))
            suspect_sentences.append(index) if isSuspect else False
            
            # compute number of chars in one suspect sentece
            dict_suspect_char_count.update({index: sum(map(len, sentences[index]))})
        
        arr_suspect_chunks = Helper.find_consecutive_numbers(suspect_sentences)
        return arr_suspect_chunks

    def compare_with_xml(self, file_item, dict_offset_index, suspect_indexes):
        """
        Compares the paragraphs detected by the algorithm 
        with the ones provided by the training corpus.
        """
        result_analizer = ResultsAnalyzer(self.corpus, file_item)
        xml_data = result_analizer.get_offset_from_xml()
        if xml_data:
            self.arr_plag_offset = [[int(x['offset']), int(x['offset'])+int(x['length'])] for x in xml_data]
            self.arr_suspect_offset = result_analizer.chunks_to_offset(dict_offset_index, suspect_indexes)

            self.arr_overlap, self.arr_suspect_overlap = result_analizer.compare_offsets(self.arr_plag_offset, self.arr_suspect_offset)

    def multi_process_array(self, arr_files, k):
        '''
        Just a wrapper so that we can multi-process.
        '''
        for f in arr_files:
            self.analize_file(f, k)

    def vectorise(self, corpus, coeficient=4, should_tokenize_corpuses=False):
        """
        Main method for vectorising the corpus. 
        """
       
        # check if tokenized is done.
        if not len(self.tokenized) and not should_tokenize_corpuses:
            print "\nImporting wikipedia dump..."
            self.tokenized = Helper.read_dump(WIKI_DUMP)
            # TODO: replace with the UnigramTagger when It will work.
            # self.tagger = Helper.read_dump(TAGGER_DUMP)
            # pdb.set_trace()

        elif not len(self.tokenized) and should_tokenize_corpuses:
            self.tokenize_corpuses(WIKI_DUMP)
            self.train_unigram_tagger(TAGGER_DUMP)
            print "\nTokenizing and training finished succesfully!"
            sys.exit()

        # Tokenizing suspicious corpus and getting most common from HUGE corpus.
        files = corpus.fileids()
        # self.suspect_corpus_tokenized = Helper.tokenize_corpus(corpus, self.stop_words, with_stop_words=True)
        
        # temporary value for k.
        # will be changed after developing a learning algorithm.
        k=coeficient
        first_half = files[:len(files)/2]
        second_half = files[len(files)/2:]
        p1 = multiprocessing.Process(target=self.multi_process_array, args=(first_half, k))
        p2 = multiprocessing.Process(target=self.multi_process_array, args=(second_half, k))

        p1.start()
        p2.start()

        p1.join()
        p2.join()
        # for file_item in files:
        #     self.analize_file(file_item, k)
            # else:
            #     print "\nNo plagiate from xml for %s" % (file_item)

            # Helper.precision(arr_suspect_chunks, dict_suspect_char_count)
        print "\n============TOTAL================="
        print "precision: ", np.mean(np.array(self.arr_mean_precision))
        print "recall: ", np.mean(np.array(self.arr_mean_recall))
        print "f1: ", np.mean(np.array(self.arr_mean_f1))
        print "\n============Missed words=========="
        print self.missed_words


 