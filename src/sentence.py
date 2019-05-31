from src.helper import Helper
from collections import Counter
from nltk.probability import FreqDist
from textstat.textstat import textstat

class SentenceAnalyser:
    '''
    Constructor
    '''
    def __init__(self, corpus, tagger, stopWords):
        self.corpus = corpus
        self.tagger = tagger
        self.stopWords = stopWords

    '''
    Main method for computing sentence feature numbers
    '''
    def compute_sentence_features(self, corpus):
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
            arr_all_sentences = []


            counter_doc_tags = Counter()

            '''
            Getting the most and least frequent 33% words
            '''
            file_words = corpus.words(fileids=file_item)
            words_in_doc = len(file_words)
            fdist_file = FreqDist(file_words)
            percentage = Helper.get_percentage(words=file_words, percentage=0.33)

            most_freq = fdist_file.most_common(percentage)
            least_freq = fdist_file.most_common()[-percentage:]


            # enumerating the paragraphs from the document and evaluating them
            for index, sentence in enumerate(corpus.sents(fileids=file_item)):
                sent_stpwrd_count = 0;
                sent_syllable_count = 0;
                sent_words_count = 0
                sent_chars_count = 0


                fdist_sentence = FreqDist(sentence)
                percentage = Helper.get_percentage(words=sentence, percentage=0.65)
                if len(sentence) < percentage:
                    most_freq_sent = fdist_sentence.most_common(percentage)
                    least_freq_sent = fdist_sentence.most_common()[-percentage]
                else:
                    most_freq_sent = fdist_sentence.most_common()
                    least_freq_sent = fdist_sentence.most_common()

                least_common = Helper.get_intersection(least_freq, least_freq_sent)
                most_common = Helper.get_intersection(most_freq, most_freq_sent)

                # calculating percentage of rare words from the sentence
                # that appear also in the document.
                # least_freq_percentage = 100*least_freq_sent/float(least_freq)
                # most_common_percentage = 100*most_freq_sent/float(most_freq)

                sent_syllable_count = textstat.syllable_count(str(sentence))
                sent_chars_count = textstat.char_count(str(sentence))
                sent_stpwrd_count = len(Helper.get_intersection(sentence, self.stopWords))
                counter_sent_tags = Counter([i[1] for i in self.tagger.tag(sentence)])
                sent_words_count = len(sentence)

                '''
                Simply add the values from the sentence pos counter
                to the doc one.
                '''
                counter_doc_tags += counter_sent_tags


                dict_sent_percents = Helper.get_feature_percentage(relative_to=sent_words_count,
                    feature=dict(counter_sent_tags))

                doc_stpwrd_count += sent_stpwrd_count
                doc_chars_count += sent_chars_count

                syllable_div_words = sent_syllable_count/sent_words_count
                chars_div_words = sent_chars_count / sent_words_count

                dict_sent_percents.update({
                    'syllable_div_words': syllable_div_words,
                    'chars_div_words': chars_div_words,
                    'sent_chars_count': sent_chars_count,
                    'sent_words_count': sent_words_count,
                    'most_freq_sent': most_freq_sent,
                    'least_freq_sent': least_freq_sent
                })

                arr_all_sentences.append({
                    'sentence_number': str(index),
                    'feature_percents': dict_sent_percents
                })
                # print("\n\===============Sentence "+str(index)+"===============\n")
                # print(arr_all_sentences)

            dict_doc_percents = Helper.get_feature_percentage(relative_to=words_in_doc,
            feature=dict(counter_doc_tags))

            dict_doc_percents.update({
                'doc_char_count': doc_chars_count
            })

            arr_synthesis.append({
                'id': file_item,
                'dict_doc_percents': dict_doc_percents,
                'word_count': words_in_doc,
                'sentence_count': len(arr_all_sentences),
                'arr_all_sentences': arr_all_sentences,
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
    def classify_chunks_sentence(self, feature_dict, corpus):

        factor1 = 8.48

        # iterating documents synthesis
        for index, item in enumerate(feature_dict):
            doc_detected_words=0

            #getting the item[dict_doc_percents]
            document_percents = item["dict_doc_percents"]

            # iterating through the paragraphs synthesis -> item['arr_all_paragraphs']
            for sentence in item["arr_all_sentences"]:
                # paragraph['feature_percents']
                sent_percents = sentence["feature_percents"]
                if not sent_percents.has_key("VB_percentage"):
                    sent_percents.update({
                        "VB_percentage": 0
                    })

                if len(sent_percents["least_freq_sent"]) > (len(item["least_freq"]) + factor1) or \
                len(sent_percents["least_freq_sent"]) < (len(item["least_freq"]) - factor1):
                    sentence.update({
                        'plagiarized_sentence': True
                    })
                    doc_detected_words += len(sent_percents["least_freq_sent"])
                else:
                    sentence.update({
                        'plagiarized_sentence': False
                    })
                ratio = doc_detected_words/float(item["word_count"])
                print("\nSentenceDocument "+str(index)+":\nRatio: " + str(ratio))
            if ratio > 5:
                item.update({
                    "plagiarized_doc": True,
                    "plagiarised_ratio": ratio
                })

        return feature_dict
