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
from sacremoses import MosesDetokenizer
from nltk import sent_tokenize, word_tokenize
from src.results_analyzer import ResultsAnalyzer
from nltk.corpus import movie_reviews, abc, brown, gutenberg, reuters, inaugural

class VectorAnaliser:
    k = 4
    """
    Constructor
    """
    def __init__(self, corpus, stop_words):
        self.corpus = corpus
        # self.tagger = tagger
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

    def feature_extraction(self, sentences, most_common_word_freq, suspicious_freq_dist):
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
        toReturn = []
        

        flat_sent = flatten(sentences)
        fdist = FreqDist(flat_sent)
        hapax = np.true_divide(len(fdist.hapaxes()), len(flat_sent))

        # computing array for all POS stored in objects
        # each object represents a sentence.
        arr_tagged_pos_per_sent = self.compute_POS(sentences)

        # iterating sentences and then words in them
        for index, words in enumerate(sentences):
            for word in words:
                word_freq = suspicious_freq_dist[word]
                if word not in  suspicious_freq_dist.keys():
                    continue
                item_index = suspicious_freq_dist.keys().index(word)

                # computing number of occurances 
                awf[item_index] = log(float(most_common_word_freq)/word_freq) / log(2)
                pcf[item_index] = fdist[word] if word in string.punctuation else 0
                stp[item_index] = fdist[word] if word in self.stop_words else 0
                pos[item_index] = arr_tagged_pos_per_sent[index][word]

                # if it's a pronoun, then we take them into account
                if arr_tagged_pos_per_sent[index][word] == 3.0 or arr_tagged_pos_per_sent[index][word] == 3.5:
                    prn[item_index] = 1
                awl.append(len(word))
                asl[index] += len(word)
            awps[index] = len(words)

            
        awf = Helper.normalize_vector([awf])
        pcf = Helper.normalize_vector([pcf])
        stp = Helper.normalize_vector([stp])
        pos = Helper.normalize_vector([pos])
        prn = Helper.normalize_vector([prn])

        toReturn.append(np.average(awl))
        toReturn.append(np.average(asl))
        toReturn.append(np.average(awps))
        toReturn.append(hapax)
        toReturn.extend(awf)
        toReturn.extend(pcf)
        toReturn.extend(stp)
        toReturn.extend(pos)
        toReturn.extend(prn)
        # return np.array(Helper.normalize_vector([toReturn]))

        return Helper.normalize_vector([toReturn])

    def compute_POS(self, sentences):
        """
        Return Array with dict per sent of pos tokenized sentences values.
        """
        arr_pos=[]
        for sentence in sentences:
            # TODO: change to use this instead of default pos_tag
            # tagged_sent = self.tagger.tag(sentence)
            
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
            dict_pos[word] = Helper.switch_pos(pos)
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
        @return: TODO: number of correct detected chars.
        """
        result_analizer = ResultsAnalyzer(self.corpus, file_item)
        xml_data = result_analizer.get_offset_from_xml()
        if xml_data:
            # actual_plagiarised_passages = result_analizer.get_plagiarised(xml_data)
            # detected_plagiarised_passages = self.md.detokenize(result_analizer.chunks_to_passages(dict_offset_index, suspect_indexes))
            self.arr_plag_offset = [[int(x['offset']), int(x['offset'])+int(x['length'])] for x in xml_data]
            self.arr_suspect_offset = result_analizer.chunks_to_offset(dict_offset_index, suspect_indexes)


            self.arr_overlap, self.arr_suspect_overlap = result_analizer.compare_offsets(self.arr_plag_offset, self.arr_suspect_offset)


    def vectorise(self, corpus, coeficient=4, should_tokenize_corpuses=False):
        """
        Main method for vectorising the corpus. 
        @param corpus:
        """
        file_name = "tokenized.pickle"
       
        # check if tokenized is done.
        if not len(self.tokenized) and not should_tokenize_corpuses:
            self.tokenized = Helper.read_dump(file_name)
            # dist_husige_token = FreqDist(self.tokenized)
        elif not len(self.tokenized) and should_tokenize_corpuses:
            self.tokenize_corpuses(file_name)

        # Tokenizing suspicious corpus and getting most common from HUGE corpus.
        files = corpus.fileids()
        # self.suspect_corpus_tokenized = Helper.tokenize_corpus(corpus, self.stop_words, with_stop_words=True)
        
        # Most common word in a big corpus.
        most_common_word_freq = FreqDist(self.tokenized).most_common(1)[0][1]

        # temporary value for k.
        # will be changed after developing a learning algorithm.
        k=coeficient
        arr_mean_precision = []
        arr_mean_recall = []
        arr_mean_f1 = []
        for file_item in files:
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

            suspicious_freq_dist = Helper.tokenize_file(corpus, self.stop_words, file_item, True)

            # tokenizing the words from the sentences 
            sentences = [word_tokenize(sent) for sent in sent_tokenize(corpus.raw(fileids=file_item))]
            
            # computing the document mean vector
            doc_mean_vector = self.feature_extraction(sentences, most_common_word_freq, suspicious_freq_dist)
            print "\n==========\nanalizing %s" % (file_item)
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
                toAppend = self.feature_extraction(arr_sentences, most_common_word_freq, suspicious_freq_dist)
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
                arr_mean_recall.append(1)
                arr_mean_precision.append(1)
                arr_mean_f1.append(1)
                print "\n%s precision: " % (file_item), 1
                print "%s recall: " % (file_item), 1
                print "%s f1: " % (file_item), 1
            else: 
                recall = Helper.precision(self.arr_overlap, self.arr_plag_offset)
                precision = Helper.recall(self.arr_suspect_overlap, self.arr_suspect_offset)
                f1 = Helper.granularity_f1(precision, recall, self.arr_overlap)

                arr_mean_recall.append(recall)
                arr_mean_precision.append(precision)
                arr_mean_f1.append(f1)
            
                print "\n%s precision: " % (file_item), precision
                print "%s recall: " % (file_item), recall
                print "%s f1: " % (file_item), f1
            # else:
            #     print "\nNo plagiate from xml for %s" % (file_item)

            # Helper.precision(arr_suspect_chunks, dict_suspect_char_count)
        pdb.set_trace()
        print "\n============TOTAL================="
        print "precision: ", np.mean(np.array(arr_mean_precision))
        print "recall: ", np.mean(np.array(arr_mean_recall))
        print "f1: ", np.mean(np.array(arr_mean_f1))


 