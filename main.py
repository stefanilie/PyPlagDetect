import os
import re
import nltk
import config
import string
import pprint

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict, treebank
from nltk.tag import UnigramTagger
from nltk.probability import FreqDist

from textstat.textstat import textstat

from compiler.ast import flatten

from collections import Counter

# load the resources


'''
Downloads all of the nltk resources.
'''
def downloadNLTKResources():
    nltk.download('all')


'''
Returns the exact number of words based on the percentage needed.
@param words - [array of strings] words in analysed structure.
@param percentage - [double] fraction of resource that needs to be extracted.
@return [int] exact number of words that have to be taken into account.
'''
def get_percentage(words, percentage):
    return int(percentage * len(words))


'''
Intersection of two provided lists.
@param list1 - [list]
@param list2 - [list]
@return [list] intersection of the two lists.
'''
def get_intersection(list1, list2):
    list1 = flatten(list1)
    list2 = flatten(list2)

    return list(set(list1).intersection(set(list2)))

'''
Method obtained from https://github.com/ypeels/nltk-book/blob/master/exercises/2.21-syllable-count.py
Calculates syllable count for the provided word.
@param word - string representing the word.
\=======================DEPRECATED========================/
'''
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
def get_feature_percentage(relative_to, feature):
    dict_reply = {}
    for key, value in feature.iteritems():
        dict_reply.update({str(key)+"_percentage": 100*value/float(relative_to)})
    return dict_reply


'''
Main method for computing paragraph feature numbers
'''
def compute_paragraph_features(corpus):
    print "\nStarting to analyse the corpus"
    files = corpus.fileids()
    arr_synthesis = []

    '''
    First we iterate over all the files we need to analyse using the
    PlaintextCorpusReader class.
    '''
    for file_item in files:
        doc_stpwrd_count = 0
        doc_chars_count = 0
        words_in_doc = 0
        arr_all_paragraphs = []


        counter_doc_tags = Counter()

        '''
        Getting the most and least frequent 24% words
        '''
        file_words = corpus.words(fileids=file_item)
        words_in_doc = len(file_words)
        fdist_file = FreqDist(file_words)
        percentage = get_percentage(words=file_words, percentage=0.24)

        most_freq = fdist_file.most_common(percentage)
        least_freq = fdist_file.most_common()[-percentage:]

        # enumerating the paragraphs from the document and evaluating them
        for index, paragraph in enumerate(corpus.paras(fileids=file_item)):
            para_stpwrd_count = 0;
            para_syllable_count = 0;
            para_words_count = 0
            para_chars_count = 0


            fdist_paragraph = FreqDist(flatten(paragraph))
            percentage = get_percentage(words=paragraph, percentage=0.66)
            most_freq_para = fdist_paragraph.most_common(percentage)
            least_freq_para = fdist_paragraph.most_common()[-percentage]

            least_common = get_intersection(least_freq, least_freq_para)
            most_common = get_intersection(most_freq, most_freq_para)

            para_syllable_count = textstat.syllable_count(str(flatten(paragraph)))
            para_chars_count = textstat.char_count(str(flatten(paragraph)))

            for sentence in paragraph:
                para_words_count += len(sentence)

                counter_para_tags = Counter([i[1] for i in tagger.tag(sentence)])
                para_stpwrd_count += len(get_intersection(sentence, stopWords))

                '''
                Simply add the values from the paragraph pos counter
                to the doc one.
                '''
                counter_doc_tags += counter_para_tags


            dict_para_percents = get_feature_percentage(relative_to=para_words_count,
                feature=dict(counter_para_tags))

            doc_stpwrd_count += para_stpwrd_count
            doc_chars_count += para_chars_count

            syllable_div_words = para_syllable_count/para_words_count
            chars_div_words = para_chars_count / para_words_count

            dict_para_percents.update({
                'syllable_div_words': syllable_div_words,
                'chars_div_words': chars_div_words,
                'para_chars_count': para_chars_count,
                'para_words_count': para_words_count
            })

            arr_all_paragraphs.append({
                'paragraph_number': str(index),
                'feature_percents': dict_para_percents
            })
            print "\n\===============Paragraph "+str(index)+"===============\n"
            print arr_all_paragraphs

        dict_doc_percents = get_feature_percentage(relative_to=words_in_doc,
        feature=dict(counter_doc_tags))

        dict_doc_percents.update({
            'doc_char_count': doc_chars_count
        })

        arr_synthesis.append({
            'id': file_item,
            'dict_doc_percents': dict_doc_percents,
            'word_count': words_in_doc,
            'paragraph_count': len(arr_all_paragraphs),
            'arr_all_paragraphs': arr_all_paragraphs
        })

    print "\n\===============Data after paraghraph analisys===============\n"
    pretty_printer.pprint(arr_synthesis)
    return arr_synthesis

def classify_chinks_paragraph(feature_dict, corpus):

    factor1 = 6.71497005988024
    factor2 = 5.755688622754491
    factor3 = 0.5915568862275449
    factor4 = 10.68

    # iterating documents synthesis
    for item in feature_dict:
        doc_detected_words=0

        #getting the item[dict_doc_percents]
        document_percents = item["dict_doc_percents"]

        # iterating through the paragraphs synthesis -> item['arr_all_paragraphs']
        for paragraph in item["arr_all_paragraphs"]:
            # paragraph['feature_percents']
            para_percents = paragraph["feature_percents"]
            if not para_percents.has_key("VB_percentage"):
                para_percents.update({
                    "VB_percentage": 0
                })

            

            if para_percents["NN_percentage"] > (document_percents["NN_percentage"] + factor1) or \
             para_percents["NN_percentage"] < (document_percents["NN_percentage"] - factor1) or \
             para_percents["VB_percentage"] > (document_percents["VB_percentage"] + factor2) or \
             para_percents["VB_percentage"] < (document_percents["VB_percentage"] - factor2) or \
             para_percents["para_chars_count"] > (document_percents["doc_char_count"] + factor3) or \
             para_percents["para_chars_count"] < (document_percents["doc_char_count"] - factor3):
                paragraph.update({
                    'plagiarized_paragraph': True
                })
                # 6 para_percents["para_words_count"]
                doc_detected_words += para_percents["para_words_count"]
            else:
                paragraph.update({
                    'plagiarized_paragraph': False
                })
        ratio = doc_detected_words*100/float(item["word_count"])
        if ratio > 5:
            item.update({
                "plagiarized_doc": True,
                "plagiarised_ratio": ratio
            })

    print "\n\===============Data after feature classification===============\n"
    pretty_printer.pprint(feature_dict)
    return feature_dict


'''
This has to be done....
'''
def compute_TF_IDF(term, document):
    tf = computeTF(term, document)
    idf = computeIDF(term, corpus)
    return tf * idf

# def computeTF(term, document):


# setting the PATH
os.chdir(config.PATH)

# initialising the corpus reader to the docs path
corpusReader = PlaintextCorpusReader(config.PATH, '.*\.txt')
# downloadNLTKResources()

# setting stopwords
stopWords = set(stopwords.words('english'))

cmdict = cmudict.dict()

pretty_printer = pprint.PrettyPrinter(indent=4)

# Training a unigram part of speech tagger
train_sents = treebank.tagged_sents()[:5000]
tagger = UnigramTagger(train_sents)
feature_arr = compute_paragraph_features(corpus=corpusReader)
feature_arr = classify_chinks_paragraph(feature_dict=feature_arr, corpus=corpusReader)

for item in feature_arr:
    if "plagiarized_doc" in item:
        print "\nDocument is plagiarised with a ratio of:"+str(item["plagiarised_ratio"])

# term-frequency of a word (tf) = 1 + log(frequency of word in a document)
# inverse-document-frequency of a word (idf) = log(Total Number of Documents in the Corpus / Number of Documents in which the word appears)
# tf-idf = tf * idf
