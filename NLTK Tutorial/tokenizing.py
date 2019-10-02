import pdb
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
example_text = "Hello there, Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is pink-ish blue. You shouldn't eat cartboard."

# print(sent_tokenize(example_text))
#
# print(word_tokenize(example_text))

# for i in word_tokenize(example_text):
#     print(i)

# Pre-compute number of chunks to emit
array = word_tokenize(example_text)
windows=[]
numOfChunks = ((len(array)-4)/1)+1

# Do the work
for i in range(0,numOfChunks*1,1):
    windows.append(array[i:i+4])

pdb.set_trace()