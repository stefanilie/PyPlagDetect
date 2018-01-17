import os
import re
import nltk
import config
import string

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict, treebank
from nltk.tag import UnigramTagger
from nltk.probability import FreqDist

from compiler.ast import flatten

# load the resources


def downloadNLTKResources():
    nltk.download('all')


def get_percentage(words, percentage):
    return int(percentage * len(words))

def get_intersection(list1, list2):
    list1 = flatten(list1)
    list2 = flatten(list2)

    return list(set(list1).intersection(set(list2)))

'''
Method obtained from https://github.com/ypeels/nltk-book/blob/master/exercises/2.21-syllable-count.py
'''
def syllables_in_word(word):
    flat_dict = dict(cmudict.entries())
    if flat_dict.has_key(word):
        return len([ph for ph in flat_dict[word] if ph.strip(string.letters)])
    else:
        return 0

def get_feature_percentage(relative_to, feature):
    dict_reply = {}
    for key, value in feature.iteritems():
        dict_reply.update({str(key)+"_percentage": 100*value/float(relative_to)})
    return dict_reply


'''
Main method for computing paragraph feature numbers
'''
def compute_paragraph_features(corpus):
    files = corpus.fileids()
    dict_synthesis={}

    '''
    First we iterate over all the files we need to analyse using the
    PlaintextCorpusReader class.
    '''
    for file_item in files:
        doc_noun_count = 0
        doc_pnoun_count = 0
        doc_pronoun_count = 0
        doc_verb_count = 0
        doc_adverb_count = 0
        doc_adj_count= 0
        doc_stpwrd_count = 0
        doc_chars_count = 0
        words_in_doc = 0
        dict_all_paragraphs = {}


        '''
        Getting the most and least frequent 24% words
        '''
        file_words = corpus.words(fileids=file_item)
        words_in_doc = len(file_words)
        # sents = corpus.sents(fileids=file_item)
        fdist_file = FreqDist(file_words)
        percentage = get_percentage(words=file_words, percentage=0.24)

        most_freq = fdist_file.most_common(percentage)
        least_freq = fdist_file.most_common()[-percentage:]

        for index, paragraph in enumerate(corpus.paras(fileids=file_item)):
            para_noun_count = 0
            para_pnoun_count = 0
            para_pronoun_count = 0
            para_verb_count = 0
            para_adverb_count = 0
            para_adj_count= 0
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

            for sentence in paragraph:
                para_words_count += len(sentence)

                for word in sentence:
                    para_chars_count += len(word)
                    para_syllable_count += syllables_in_word(word=word)
                    if word in stopWords:
                        para_stpwrd_count += 1
                for tok, tag in tagger.tag(sentence):
                    if str(tag) == "NN" or str(tag) == "NNS":
                        para_noun_count += 1
                    elif str(tag) == "NNP" or str(tag) == "NNPS":
                        para_pnoun_count += 1
                    elif str(tag) == "PRP" or str(tag) == "PRP$":
                        para_pronoun_count += 1
                    elif re.match(r"^VB[A-Z]$", str(tag)) or str(tag) == "VB":
                        para_verb_count += 1
                    elif str(tag) == "RB" or re.match(r"^RB[A-Z]", str(tag)):
                        para_adverb_count += 1
                    elif str(tag) == "JJ" or re.match(r"^JJ[A-Z]", str(tag)):
                        para_adj_count += 1

            dict_para_features = {
                'para_adj': para_adj_count,
                'para_adverb': para_adverb_count,
                'para_noun': para_noun_count,
                'para_verb': para_verb_count,
                'para_pnoun': para_pnoun_count,
                'para_pronoun': para_pronoun_count,
                'para_stpwrd': para_stpwrd_count
            }
            dict_para_percents = get_feature_percentage(relative_to=para_words_count, feature=dict_para_features)

            doc_adj_count += para_adj_count
            doc_noun_count += para_noun_count
            doc_verb_count += para_verb_count
            doc_pnoun_count += para_pnoun_count
            doc_pronoun_count += para_pronoun_count
            doc_adverb_count += para_adverb_count
            doc_stpwrd_count += para_stpwrd_count
            doc_chars_count += para_chars_count

            syllable_div_words = para_syllable_count/para_words_count
            chars_div_words = para_chars_count / para_words_count

            dict_para_features.update({
                'syllable_div_words': syllable_div_words,
                'chars_div_words': chars_div_words,
                'para_chars_count': para_chars_count,
                'para_words_count': para_words_count
            })

            dict_all_paragraphs.update({
                'paragraph_number': str(index),
                'feature_percents': dict_para_percents
            })
            print "\n\===============Paragraph "+str(index)+"===============\n"
            print dict_all_paragraphs


        dict_doc_features = {
            'doc_adj': doc_adj_count,
            'doc_adverb': doc_adverb_count,
            'doc_noun': doc_noun_count,
            'doc_verb': doc_verb_count,
            'doc_pnoun': doc_pnoun_count,
            'doc_pronoun': doc_pronoun_count,
            'doc_stpwrd': doc_stpwrd_count,
            'doc_chars': doc_chars_count
        }

        dict_doc_percents = get_feature_percentage(relative_to=words_in_doc, feature=dict_doc_features)
        print dict_doc_percents
        print "\n\n"

        dict_synthesis.update({
            'id': file_item,
            'dict_doc_percents': dict_doc_percents,
            'word_count': words_in_doc,
            'paragraph_count': len(dict_all_paragraphs),
            'dict_all_paragraphs': dict_all_paragraphs
        })


    print "\n\===============dict_synthesis===============\n"
    print dict_synthesis
    return dict_synthesis

def classify_chinks_paragraph(feature_dict, corpus):

    factor1 = 6.71497005988024
    factor2 = 5.755688622754491
    factor3 = 0.5915568862275449
    factor4 = 10.68

    # iterating documents synthesis
    for item in feature_dict:
        doc_detected_words=0
        # schimba aici sa primeasca inturi...
        # TypeError: string indices must be integers, not str
        document_percents = item[0]
        # iterating through the paragraphs synthesis
        for paragraph in item['dict_all_paragraphs']:
            para_percents = paragraph['feature_percents']
            if para_percents['para_noun_percentage'] > (document_percents['doc_noun_percentage'] + factor1) or \
             para_percents['para_noun_percentage'] < (document_percents['doc_noun_percentage'] - factor1) or \
             para_percents['para_verb_percentage'] > (document_percents['doc_verb_percentage'] + factor2) or \
             para_percents['para_verb_percentage'] < (document_percents['doc_verb_percentage'] - factor2) or \
             para_percents['para_chars_count'] > (document_percents['doc_chars_count'] + factor3) or \
             para_percents['para_chars_count'] < (document_percents['doc_chars_count'] - factor3):
                paragraph.update({
                    'plagiarized': True
                })
                doc_detected_words += para_percents["para_words_count"]
            else:
                paragraph.update({
                    'plagiarized': False
                })
        ratio = doc_detected_words*100/float(item["word_count"])
    print feature_dict
    return feature_dict


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
# print(cmudict.entries()[653:659])

# Training a unigram part of speech tagger
train_sents = treebank.tagged_sents()[:5000]
tagger = UnigramTagger(train_sents)
feature_dict = compute_paragraph_features(corpus=corpusReader)
classify_chinks_paragraph(feature_dict=feature_dict, corpus=corpusReader)


# term-frequency of a word (tf) = 1 + log(frequency of word in a document)
# inverse-document-frequency of a word (idf) = log(Total Number of Documents in the Corpus / Number of Documents in which the word appears)
# tf-idf = tf * idf
