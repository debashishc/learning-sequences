
# Load Larger LSTM network and generate text
from __future__ import print_function
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

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

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# load the network weights
filename = "weights-improvement-39-1.3837-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

text_dict = dict()
for ix in range(10000):
	if ix%10 == 0:
		print('Iteration: ', ix)

	# pick a random seed
	start = numpy.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
	# print ("Seed text:")
	seed_text = ''.join([int_to_char[value] for value in pattern])
	# print ("\"", seed_text, "\"")

	# generate characters
	generated_text = ''
	for i in range(100):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = int_to_char[index]
		generated_text += result
		seq_in = [int_to_char[value] for value in pattern]
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	# print("\nGenerated text: \n", generated_text)
	# print ("\nDone.")

	text_dict[ix] = (seed_text, 1)
	text_dict[ix+100] = (generated_text, 0)

print(text_dict)

import csv

# create csv file for text and score for human(1) or machine(0) generated
fieldnames = ['index', 'text', 'score']
with open('text_scores.csv', 'w', newline='') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	data = [dict(zip(fieldnames, [k, v[0], v[1]])) for k, v in text_dict.items()]
	writer.writerows(data)