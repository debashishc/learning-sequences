import nltk.data

import nltk
nltk.download('punkt')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open("got1.txt")
data = fp.read()
# print ('\n-----\n'.join(tokenizer.tokenize(data)))
print(tokenizer.tokenize(data))