
# Load Larger LSTM network and generate text
from __future__ import print_function
import sys
import numpy
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# for bigrams
from collections import Counter, defaultdict
from itertools import tee
from count_bigrams import pairwise, bigrams

NUM_GENERATED = 50 # number of sequences to be generated
print("\n********************* GENERATING ", NUM_GENERATED, " SENTENCES *********************\n")

import string
def remove_punc(sentence):
	"""Remove punctuations from a given sentence """
	return "".join(char for char in sentence if char not in string.punctuation)

# load ascii text and covert to lowercase
# TRAIN_FILE = 'input/colombiano.txt'
TRAIN_FILE = "input/alice-in-wonderland.txt"
raw_text = open(TRAIN_FILE).read()
raw_text = raw_text.lower()

# load the network weights
weights_folder = ""
# weights_folder = "large-lstm-weights/" # english
# weights_folder = "spanish-weights-bigger/" #spanish

# WEIGHT = "weights-improvement-39-1.3837-bigger.hdf5" # english
WEIGHT = "weights-improvement-20-0.9309-bigger.hdf5" # 512 unit LSTM
# WEIGHT = "weights-improvement-17-1.5762-bigger.hdf5" # spanish

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
print(X.shape)

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


weights_file = weights_folder + WEIGHT
model.load_weights(weights_file)
model.compile(loss='categorical_crossentropy', optimizer='adam')

text_dict = dict()
for ix in range(NUM_GENERATED):
	if ix%10 == 0:
		print('Generation: ', ix)

	# pick a random seed
	start = numpy.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
	print ("Seed text:")
	seed_text = ''.join([int_to_char[value] for value in pattern])
	print ("\"", seed_text, "\"")

	# generate characters
	generated_text = ''

	for i in range(300):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = int_to_char[index]
		generated_text += result
		seq_in = [int_to_char[value] for value in pattern]
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	print("\nGenerated text: \n", generated_text)
	print ("\nDone.")


	def num_spell_correctly(sentence, word_file):
		counter = 0
		correct = list()
		with open(word_file, 'r') as _file:
			data = _file.read().split('\n')
		# print("SENTENCE: ", sentence)
		for word in sentence.split():
			if remove_punc(word) in data:
				correct.append(word)
				counter += 1
		return counter

	def num_novel(sentence, word_file, train_text_file):
		counter = 0
		correct = list()
		with open(train_text_file, 'r') as _file, open(word_file, 'r') as _words:
			data = _file.read().split()
			data = [remove_punc(word) for word in data]
			all_words = _words.read().split('\n')
		# print("SENTENCE: ", sentence)
		for word in sentence.split():
			if remove_punc(word) not in data and remove_punc(word) in all_words:
				correct.append(word)
				counter += 1
		return counter, correct
	
	def num_novel_all(sentence, train_text_file):
		counter = 0
		correct = list()
		with open(train_text_file, 'r') as _file:
			data = _file.read().split()
			data = [remove_punc(word) for word in data]
		for word in sentence.split():
			if remove_punc(word) not in data:
				correct.append(word)
				counter += 1
		return counter, correct


	WORDS_FILENAME = 'words_alpha.txt'

	# print(num_spell_correctly(generated_text, WORDS_FILENAME))

	text_dict[ix] = (seed_text, 1)
	text_dict[ix+1000] = (generated_text, 0)


# # create csv file for text and score for human(1) or machine(0) generated
# fieldnames = ['index', 'text', 'score','correct spelling','percent','novel word spelled', 'novel words', 'bigrams']
# with open('text_scores_new.csv', 'w', newline='') as csvfile:
# 	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# 	writer.writeheader()
# 	data = [dict(zip(fieldnames, [k, v[0], v[1], num_spell_correctly(v[0], WORDS_FILENAME), 
#                                round((num_spell_correctly(v[0], WORDS_FILENAME)/len(v[0].split())), 2), 
# 							   num_novel(v[0], WORDS_FILENAME, TRAIN_FILE), num_novel_all(v[0], TRAIN_FILE), bigrams(v[0])]))
#          for k, v in text_dict.items()]
# 	writer.writerows(data)



if __name__ == '__main__':
	pass
