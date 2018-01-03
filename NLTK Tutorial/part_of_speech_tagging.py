import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

# nltk.download('all')

def process_content(tokenized_text):
    try:
        for sentance in tokenized_text:
            # iterating and tokenising each word from each sentance
            words = nltk.word_tokenize(sentance)
            # then for each word, tagging the POS
            tagged = nltk.pos_tag(words)

            # Chinking and Chunking
            #
            # chunkGram = r"""Chunk: {<.*>+}
            #                 }<VB.?|IN|DT>+{
            #             """
            #
            # chunkParser = nltk.RegexpParser(chunkGram)
            # chunked = chunkParser.parse(tagged)
            #
            # chunked.draw()

            # Detecting named entities.
            # If you want to see the type of the named entity, delete binary var.
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()

    except Exception as e:
        print(str(e))


# loading the corpus
sample_text = state_union.raw("2006-GWBush.txt")
train_text = state_union.raw("2005-GWBush.txt")

#initialising the tokenizer and training it on a corpus
punkt_tokenizer = PunktSentenceTokenizer(train_text)

# tokenizing the text and storing it in a var, sentence by sentence
tokenized_text = punkt_tokenizer.tokenize(sample_text)

process_content(tokenized_text)
