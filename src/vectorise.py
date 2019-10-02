import os
import sys
import time
import pickle
import pyphen
import string
import pprint
import numpy as np
from tqdm import tqdm
import scipy.stats as sc

from numpy import dot
from nltk import pos_tag
from funcy import flatten
from math import log, floor
from src.helper import Helper
from numpy.linalg import norm
from collections import Counter
from nltk.tag import UnigramTagger
from gensim.corpora import WikiCorpus
from nltk.probability import FreqDist
from sacremoses import MosesDetokenizer
from multiprocessing import Process, Manager
from nltk import sent_tokenize, word_tokenize
from src.results_analyzer import ResultsAnalyzer
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus import movie_reviews, abc, brown, gutenberg, reuters, inaugural
from src.config import WIKI_DUMP, TAGGER_DUMP, SMALL_DUMP, OANC, SUSPICIOUS_DOCUMENTS, PLOTS, WIKI_FILE_NAME

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class VectorAnaliser:
    def __init__(self, corpus, stop_words, custom_mode=False):
        """
        Constructor
        """
        self.corpus = corpus
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
        self.custom_mode = custom_mode
        self.pretty_printer = pprint.PrettyPrinter(indent=2)

    def should_tokenize(self, should_tokenize_corpuses=False):
        # check if tokenized is done.
        if not len(self.tokenized) and not should_tokenize_corpuses:
            print("\nReading wikipedia dump...")
            self.tokenized = Helper.read_dump(WIKI_DUMP)
            print("\nReading UnigramTagger dump...")
            self.tagger = Helper.read_dump(TAGGER_DUMP)

        elif not len(self.tokenized) and should_tokenize_corpuses:
            if os.path.isfile(WIKI_FILE_NAME):
                self.tokenize_corpuses(WIKI_DUMP, WIKI_FILE_NAME)
                self.train_unigram_tagger(TAGGER_DUMP)
                print("\nTokenizing and training finished succesfully!")
                sys.exit()
            else:
                print("\nThe wikipedia file dump not present in the root folder.")
                sys.exit()

    def multi_process_files(self, arr_files, k, arr_mean_precision, arr_mean_recall, arr_mean_f1, dict_file_vectors={}, kmeans=False):
        """
        Just a wrapper so that we can multi-process.
        """
        for f in arr_files:
            self.analize_file(f, k, arr_mean_precision, arr_mean_recall, arr_mean_f1, dict_file_vectors, kmeans)

    def analize_file(self, file_item, k, arr_mean_precision, arr_mean_recall, arr_mean_f1, dict_file_vectors, kmeans=False):
        """
        Analyzes file by creating windows
        and extracting window features.
        """

        # Most common word in a big corpus.
        # most_common_word_freq = FreqDist(self.tokenized).most_common(1)[0][1]
        most_common_word_freq = self.tokenized.most_common()[0][1]

        windows_total = []
        doc_mean_vector = []
        dict_all_sentences = {} # used to save all the windows sent for analysis.
        dict_offset_index = {} # used for saving the start offset and length of each window.
        offset_counter = 0
        self.arr_plag_offset = [] # [offset, length] items of real plagiarised.
        self.arr_suspect_offset = [] # [offset, length] items of suspect plagiarised.
        self.arr_suspect_overlap = [] # no. of common chars by real plag
        self.arr_cosine_similarity = [] # all cosine similarities by sentence no.
        self.mean = 0 # mean of cosine similarities.
        self.standard_deviation = 0


        suspicious_freq_dist = Helper.tokenize_file(self.corpus, self.stop_words, file_item, True)

        # tokenizing the words from the sentences 
        sentences = [word_tokenize(sent) for sent in sent_tokenize(self.corpus.raw(fileids=file_item))]
        
        """
        Computing the document mean vector
        calling feature method with all sentences. 
        """
        print("\n==========\nanalizing %s" % (file_item))
        print("\nComputing reference_vector")
        doc_mean_vector = self.feature_extraction(sentences, most_common_word_freq, suspicious_freq_dist, True)
        print("\nComputing features")
        for index, sentence in enumerate(tqdm(sentences)):
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
                arr_sentences = sentences[index-int(k/2):index+int(k/2)]                    

            # extracting features
            toAppend = self.feature_extraction(arr_sentences, most_common_word_freq, suspicious_freq_dist, False)

            # adding it to the document feature set
            windows_total.append(toAppend)
            
            # indexing all sentences for easy access
            dict_all_sentences[index] = sentence

            """
            saving the offset and length of the current sentence.
            this will be used for overlapping 
            data from xml with these found here.
            """
            new_offset = len(self.md.detokenize(sentence))
            offset_counter += new_offset
            dict_offset_index[index] = [offset_counter, new_offset]


        # compute cosine similarity for all windows plus mean
        self.compute_cosine_similarity_array(windows_total, doc_mean_vector)  

        # computing the standard deviation 
        self.standard_deviation = Helper.stddev(windows_total, self.arr_cosine_similarity, self.mean)

        # getting the intervals between which the
        # suspect passages lay
        arr_suspect_chunks = self.get_suspect_index(sentences)

        # comparing them with the real results
        self.compare_with_xml(file_item, dict_offset_index, arr_suspect_chunks)

        # if we are not just analyzing the file
        # without evaluating the algorithm
        if self.custom_mode == False:
            recall = 0
            precision = 0
            f1 = 0
            if len(self.arr_plag_offset) == 0 and len(self.arr_suspect_offset) == 0:
                arr_mean_recall.append(1)
                arr_mean_precision.append(1)
                arr_mean_f1.append(1)
                print("\n%s precision: " % (file_item), 1)
                print("%s recall: " % (file_item), 1)
                print("%s f1: " % (file_item), 1)
                print("\n%s: No plagiarism detected and none existing" % (file_item))
            else: 
                # computing scores
                precision = Helper.precision(self.arr_overlap, self.arr_plag_offset)
                recall = Helper.recall(self.arr_suspect_overlap, self.arr_suspect_offset)
                f1 = Helper.granularity_f1(precision, recall, self.arr_overlap)

                # adding them to the mean list
                arr_mean_recall.append(recall)
                arr_mean_precision.append(precision)
                arr_mean_f1.append(f1)

                # printing results for that file.
                print("\n%s precision: " % (file_item), precision)
                print("%s recall: " % (file_item), recall)
                print("%s f1: " % (file_item), f1)

        if kmeans:
            dict_file_vectors[file_item] = windows_total

    def kmeans(self, vector, file_name, K=2):
        current_directory = os.getcwd()
        os.chdir(PLOTS)
        arr = (np.array(vector))

        # mean normalization of the data . converting into normal distribution having mean=0 , -0.1<x<0.1
        sc = StandardScaler()
        x = sc.fit_transform(arr)

        # Breaking into principle components
        pca = PCA(n_components=2)
        components = (pca.fit_transform(x))
        
        # Applying kmeans algorithm for finding centroids
        kmeans = KMeans(n_clusters=K, n_jobs=-1)
        kmeans.fit_transform(components)
        print("labels: ", kmeans.labels_)
        centers = kmeans.cluster_centers_

        # lables are assigned by the algorithm if 2 clusters then lables would be 0 or 1
        lables = kmeans.labels_
        colors = ["r.", "g.", "b.", "y.", "c."]
        colors = colors[:K + 1]

        print("Creating plots for %s" % (file_name))
        for index, component in enumerate(tqdm(components)):
            plt.plot(component[0], component[1], colors[lables[index]], markersize=5)

        plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=150, linewidths=10, zorder=15)
        plt.xlabel("1st Principle Component")
        plt.ylabel("2nd Principle Component")
        title = "Styles Clusters for %s" % (file_name)
        plt.title(title)
        plt.savefig("kmeans-%s.png" % (file_name))
        plt.clf()
        os.chdir(current_directory)

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
        awd =  0 # average word diversity (no unique words/no words)
        aspw = [] #average syllable count per word
        fre = 0 # flesch reading ease 
        gre = 0 # flesch kincaid grade level
        sha_entr = [] # shanon entropy value
        toReturn = []
        

        flat_sent = list(flatten(sentences))
        fdist = FreqDist(flat_sent)

        # computing number of hapax legomena
        hapax = np.true_divide(len(fdist.hapaxes()), len(flat_sent))

        # average word diversity
        awd = np.true_divide(len(fdist), len(flat_sent))

        # computing array for all POS stored in objects
        # each object represents a sentence.
        arr_tagged_pos_per_sent = self.compute_POS(sentences)

        # iterating sentences and then words in them
        for index, words in enumerate(sentences):
            # if verbose:
            #     Helper.print_progress(index, len(sentences))
            for word in words:
                word = word.lower()
                word_freq = suspicious_freq_dist[word]
                word_freq_wiki_corpus = self.tokenized[word]

                if '-' in word and word_freq_wiki_corpus == 0:
                        word_freq_wiki_corpus = np.average([self.tokenized[i] for i in word.split('-')])

                # this coveres the cases where the word 
                # isn't part of the big wiki freq dist.
                if word_freq_wiki_corpus == 0:
                    if word not in list(self.coca_freq_dict.keys()):
                        if word not in self.missed_words:
                            self.missed_words.append(word)
                        word_freq_wiki_corpus = 1
                    else: 
                        word_freq_wiki_corpus = self.coca_freq_dict[word]
                
                if word not in list(suspicious_freq_dist.keys()):
                    # this is for some cases where 
                    # we have words that couldn't be tokenized.
                    continue
                item_index = list(suspicious_freq_dist.keys()).index(word)

                # computing window caracteristics.
                awf[item_index] = floor(log(np.true_divide(most_common_word_freq, word_freq_wiki_corpus))/log(2))
                # awf[item_index] = floor(log(np.true_divide(word_freq_wiki_corpus, word_freq))/log(2))
                pcf[item_index] = fdist[word] if word in string.punctuation else 0
                stp[item_index] = fdist[word] if word in self.stop_words else 0
                pos[item_index] = arr_tagged_pos_per_sent[index][word]

                # if it's a pronoun, then we take them into account
                if arr_tagged_pos_per_sent[index][word] == 3.0 or arr_tagged_pos_per_sent[index][word] == 3.5:
                    prn[item_index] = 1
                
                # average word length
                awl.append(len(word))

                # average sentence length
                asl[index] += len(word)

                # average syllable count per word
                aspw.append(len(self.dic.inserted(word).split('-')))

                # shanon entropy
                sha_entr.append(word_freq*log(word_freq, 2))

            # average words per sentence
            awps[index] = len(words)
        
        # flesch reading ease 
        # flesch-kincaid grade
        words = np.sum(awps)
        syllables = np.sum(aspw)
        fre, gre = Helper.compute_flesch_reading_ease(words, syllables, len(sentences))
        
        sha_entr = sc.entropy(sha_entr, None, 2)

        # normalizing vectors
        awf = Helper.normalize_vector([awf])
        pcf = Helper.normalize_vector([pcf])
        stp = Helper.normalize_vector([stp])
        pos = Helper.normalize_vector([pos])
        prn = Helper.normalize_vector([prn])

        # adding data to document data vector
        toReturn.append(np.average(awl))
        toReturn.append(np.average(asl))
        toReturn.append(np.average(awps))
        toReturn.append(np.average(aspw))
        toReturn.append(np.average(awf))
        toReturn.append(hapax)

        toReturn.append(fre)
        toReturn.append(gre)
        toReturn.append(sha_entr)
        toReturn.append(awd)
        
        toReturn.extend(awf)
        toReturn.extend(pcf)
        toReturn.extend(stp)
        toReturn.extend(pos)
        toReturn.extend(prn)

        # normalizing result
        return Helper.normalize_vector([toReturn])

    def tokenize_wikipedia(self, wiki_file_name):
        """
        Creates a pickle file for a wikipedia dump.
        File will contain the FreqDist.
        """
        i = 0
        wiki_fdist = FreqDist()
        wiki = WikiCorpus(wiki_file_name)
        for text in wiki.get_texts():
            text_fdist = FreqDist(text)
            i = i + 1
            wiki_fdist += text_fdist
            if (i % 10000 == 0):
                print('Processed ' + str(i) + ' articles')
        Helper.create_dump(wiki_fdist, WIKI_DUMP)

    def tokenize_corpuses(self, file_name, wiki_file_name):
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
        self.tokenized += self.tokenize_wikipedia(wiki_file_name)
        Helper.create_dump(self.tokenized, file_name)

    def train_unigram_tagger(self, file_name):
        """
        Trains an unigram tagger based on OANC.
        Exports data to a dump file in the dumps folder.
        """
        toTag = []
        files = self.corpus.fileids()
        print("\nExtracting words from files...")
        for index, file_item in enumerate(tqdm(files)):
            paras = self.corpus.paras(file_item)
        
            try:
                for sentences in paras:
                    st = list(flatten(sentences))
                    training_tags = []
                    for word in st:
                        if '_' in word:
                            if len(word.split('_')) > 2:
                                continue
                            w, pos = word.split('_')
                            if w != '' and pos != '':
                                training_tags.append((w, pos))
                    if len(training_tags) > 0:
                        toTag.append(training_tags)
            except:
                print("Error at file %s" %(file_item))

        print("\Training tagger..." )
        pos_tagger = UnigramTagger(toTag)

        print("\nCreating dump in unigram_tagger.pickle...")
        Helper.create_dump(pos_tagger, file_name)

    def compute_POS(self, sentences):
        """
        Return Array with dict per sent of pos tokenized sentences values.
        """
        arr_pos=[]
        for sentence in sentences:
            # TODO: change to use this instead of default pos_tag
            tagged_sents = self.tagger.tag(sentence)
            # tagged_sents = pos_tag(sentence)

            # check if tagged_sents works well
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

        print("\nComputing cosine_similarity")
        for window in tqdm(windows):
            # Helper.print_progress(index, len(windows))
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
            
            # print("index: %s, is suspect: %s" %(str(index), str(isSuspect)))
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
        result_analizer = ResultsAnalyzer(self.corpus, file_item, self.custom_mode)
        xml_data = result_analizer.get_offset_from_xml()
        if xml_data != []:
            self.arr_plag_offset = [[int(x['offset']), int(x['offset'])+int(x['length'])] for x in xml_data]
            self.arr_suspect_offset = result_analizer.chunks_to_offset(dict_offset_index, suspect_indexes)

            self.arr_overlap, self.arr_suspect_overlap = result_analizer.compare_offsets(self.arr_plag_offset, self.arr_suspect_offset)
        elif xml_data == [] and self.custom_mode ==True:
            passages = result_analizer.chunks_to_passages(dict_offset_index, suspect_indexes)
            print("\nPossible plagiarised passages for %s:" % (file_item))
            self.pretty_printer.pprint(passages)
           
    def vectorise(self, corpus, coeficient=6, multiprocessing=False):

        """
        Main method for vectorising the corpus. 
        @param corpus: PlainTextCorpusReade that will 
        handle reading suspect files.
        @param coeficient: window size
        """
        
        # monitoring execution time for performance reasons
        start_time = time.time()

        # Tokenizing suspicious corpus and getting most common from HUGE corpus.
        files = corpus.fileids()
        
        # depending on custom_mode, it loads the dump
        # or it tokenizes and trains.
        self.should_tokenize()

        k=coeficient

        if multiprocessing:
            # initialising multi-threading
            manager = Manager()

            # spliting the files into two
            first_half = files[:len(files)/2]
            second_half = files[len(files)/2:]
            
            # creating shared lists
            # multi-processing doesn't support global variables. 
            arr_mean_precision = manager.list()
            arr_mean_recall = manager.list()
            arr_mean_f1 = manager.list()
            dict_file_vectors = manager.dict()

            # defining processes
            p1 = Process(target=self.multi_process_files, args=(first_half, k, arr_mean_precision, arr_mean_recall, arr_mean_f1, dict_file_vectors))
            p2 = Process(target=self.multi_process_files, args=(second_half, k, arr_mean_precision, arr_mean_recall, arr_mean_f1, dict_file_vectors))
        
            # starting processes
            p1.start()
            p2.start()

            # mantaining processes
            p1.join()
            p2.join()
            
            # printing results
            print("\n=================TOTAL=================")
            print("precision: ", np.mean(np.array(arr_mean_precision)))
            print("recall: ", np.mean(np.array(arr_mean_recall)))
            print("f1: ", np.mean(np.array(arr_mean_f1)))

            if (not p1.is_alive() and not p2.is_alive()) and dict_file_vectors != {}:
                print("\n=================K-Means=================")
                for file_item in files:
                    if file_item in dict_file_vectors:
                        self.kmeans(dict_file_vectors[file_item], file_item)

            # printing execution time
            print("--- Execution time: %s seconds ---" % (time.time() - start_time))

        else:
            arr_mean_precision = []
            arr_mean_recall = []
            arr_mean_f1 = []
            
            self.multi_process_files(files, k, arr_mean_precision, arr_mean_recall, arr_mean_f1, kmeans=True)

            if self.custom_mode == False:
            # printing results
                print("\n=================TOTAL=================")
                print("precision: ", np.mean(np.array(arr_mean_precision)))
                print("recall: ", np.mean(np.array(arr_mean_recall)))
                print("f1: ", np.mean(np.array(arr_mean_f1)))

            # printing execution time
            print("--- Execution time: %s seconds ---" % (time.time() - start_time))

