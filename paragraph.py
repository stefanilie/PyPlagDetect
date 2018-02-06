from helper import Helper
from collections import Counter
from compiler.ast import flatten
from nltk.probability import FreqDist
from textstat.textstat import textstat


class ParagraphAnalyser:
    '''
    Constructor
    '''
    def __init__(self, corpus, tagger, stopWords):
        self.corpus = corpus
        self.tagger = tagger
        self.stopWords = stopWords

    '''
    Main method for computing paragraph feature numbers
    '''
    def compute_paragraph_features(self, corpus):
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
            percentage = Helper.get_percentage(words=file_words, percentage=0.24)

            most_freq = fdist_file.most_common(percentage)
            least_freq = fdist_file.most_common()[-percentage:]

            # enumerating the paragraphs from the document and evaluating them
            for index, paragraph in enumerate(corpus.paras(fileids=file_item)):
                para_stpwrd_count = 0;
                para_syllable_count = 0;
                para_words_count = 0
                para_chars_count = 0


                fdist_paragraph = FreqDist(flatten(paragraph))
                percentage = Helper.get_percentage(words=paragraph, percentage=0.66)
                most_freq_para = fdist_paragraph.most_common(percentage)
                least_freq_para = fdist_paragraph.most_common()[-percentage]

                least_common = Helper.get_intersection(least_freq, least_freq_para)
                most_common = Helper.get_intersection(most_freq, most_freq_para)

                # calculating percentage of rare words from the paragraph
                # that appear also in the document.
                # least_freq_percentage = 100*least_freq_para/float(least_freq)
                # most_common_percentage = 100*most_freq_para/float(most_freq)

                para_syllable_count = textstat.syllable_count(str(flatten(paragraph)))
                para_chars_count = textstat.char_count(str(flatten(paragraph)))

                for sentence in paragraph:
                    para_words_count += len(sentence)

                    counter_para_tags = Counter([i[1] for i in self.tagger.tag(sentence)])
                    para_stpwrd_count += len(Helper.get_intersection(sentence, self.stopWords))

                    '''
                    Simply add the values from the paragraph pos counter
                    to the doc one.
                    '''
                    counter_doc_tags += counter_para_tags


                dict_para_percents = Helper.get_feature_percentage(relative_to=para_words_count,
                    feature=dict(counter_para_tags))

                doc_stpwrd_count += para_stpwrd_count
                doc_chars_count += para_chars_count

                syllable_div_words = para_syllable_count/para_words_count
                chars_div_words = para_chars_count / para_words_count

                dict_para_percents.update({
                    'syllable_div_words': syllable_div_words,
                    'chars_div_words': chars_div_words,
                    'para_chars_count': para_chars_count,
                    'para_words_count': para_words_count,
                    'most_freq_para': most_freq_para,
                    'least_freq_para': least_freq_para
                })

                arr_all_paragraphs.append({
                    'paragraph_number': str(index),
                    'feature_percents': dict_para_percents
                })
                # print "\n\===============Paragraph "+str(index)+"===============\n"
                # print arr_all_paragraphs

            dict_doc_percents = Helper.get_feature_percentage(relative_to=words_in_doc,
            feature=dict(counter_doc_tags))

            dict_doc_percents.update({
                'doc_char_count': doc_chars_count
            })

            arr_synthesis.append({
                'id': file_item,
                'dict_doc_percents': dict_doc_percents,
                'word_count': words_in_doc,
                'paragraph_count': len(arr_all_paragraphs),
                'arr_all_paragraphs': arr_all_paragraphs,
                'most_freq': most_freq,
                'least_freq': least_freq

            })

        return arr_synthesis



    '''
    Analyses the results calculated by compute_paragraph_features,
    and compares them to the data characteristic to the doc.
    @param feature_dict [dict] data from the compute method
    @param corpus [PlaintextCorpusReader]
    @return [dict] mostly same as feature_dict, but containing two extra fields:
    "plagiarized_doc" and "plagiarised_ratio"
    '''
    def classify_chunks_paragraph(self, feature_dict, corpus):

        factor1 = 6.71497005988024
        factor2 = 5.755688622754491
        factor3 = 0.5915568862275449
        factor4 = 10.68

        # iterating documents synthesis
        for index, item in enumerate(feature_dict):
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
                if not para_percents.has_key("NN_percentage"):
                    para_percents.update({
                        "NN_percentage": 0
                    })

                if para_percents["NN_percentage"] > (document_percents["NN_percentage"] + factor1) or \
                 para_percents["NN_percentage"] < (document_percents["NN_percentage"] - factor1) or \
                 para_percents["VB_percentage"] > (document_percents["VB_percentage"] + factor2) or \
                 para_percents["VB_percentage"] < (document_percents["VB_percentage"] - factor2) or \
                 para_percents["para_chars_count"] > (document_percents["doc_char_count"] + factor3) or \
                 para_percents["para_chars_count"] < (document_percents["doc_char_count"] - factor3) or \
                 len(para_percents["least_freq_para"]) > (len(document_percents["least_freq"]) + factor4) or \
                 len(para_percents["least_freq_para"]) < (len(document_percents["least_freq"]) - factor4):
                    paragraph.update({
                        'plagiarized_paragraph': True
                    })
                    doc_detected_words += len(para_percents["least_freq_para"])
                else:
                    paragraph.update({
                        'plagiarized_paragraph': False
                    })
            ratio = doc_detected_words*100/float(item["word_count"])
            # print "\nParagraphDocument "+str(index)+":\nRatio: " + str(ratio)
            if ratio > 5:
                item.update({
                    "plagiarized_doc": True,
                    "plagiarised_ratio": ratio
                })

        return feature_dict
