import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
example_text = "Hello there, Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is pink-ish blue. You shouldn't eat cartboard."

# print(sent_tokenize(example_text))
#
# print(word_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)
