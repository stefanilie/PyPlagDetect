from compiler.ast import flatten
from nltk import word_tokenize
from nltk.corpus.reader.tagged import CategorizedTaggedCorpusReader
from nltk.corpus.util import LazyCorpusLoader

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
            dict_reply.update({str(key)+"_percentage": 100*value/float(relative_to)})
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
        return 1 + math.log(tokenized_document.count(term))

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
        return math.log(len(tokenized_document/doc_number))


    '''
    Computes the TF-IDF value.
    @param term - [string] the term to calculate the tf-idf value.
    @param document - [list of string] document or array of docs that needs to be
    calculated.
    @return [int] - value of the computed Tf-Idf
    '''
    @staticmethod
    def compute_TF_IDF(term, document):
        tf = computeTF(term, document)
        idf = computeIDF(term, document)
        return tf * idf

    '''
    Tokenizes all files from a corpus.
    @param corpus - [nltk.corpus]
    @param stopWords - [nltk.stopWords] array containing all eng stopwords.
    @return [list of strings] - tokenized array for all the corpus docs.
    '''
    @staticmethod
    def tokenize_corpus(corpus, stopWords, cathegorized=False):
        print "==========="
        print "Tokenizeing ", corpus
        tokenized = []
        # TODO: make case for corpus is brown (cathegorised).
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
                print "on the right path"
                tokenized += corpus.words()
            else:
                tokenized += word_tokenize(corpus.raw())       
        tokenized = Helper.get_difference(tokenized, stopWords)
        return tokenized
