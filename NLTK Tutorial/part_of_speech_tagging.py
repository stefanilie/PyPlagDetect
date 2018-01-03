import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

nltk.download('state_union')
nltk.download('averaged_perceptron_tagger')

def process_content(tokenized_text):
    try:
        for sentance in tokenized_text:
            # iterating and tokenising each word from each sentance
            words = nltk.word_tokenize(sentance)
            # then for each word, tagging the POS
            tagged = nltk.pos_tag(words)

            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP><NN>?}"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            chunked[].draw()

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
