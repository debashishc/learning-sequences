# LSTM Network to Generate Text
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
def get_raw_text(filename):
	"""
	Read file into lower case text
	"""
	with open(filename) as f:
		raw_text = f.read()
		raw_text = raw_text.lower()
	return raw_text


### CON: model not considering upper case letters

def get_char_to_int(raw_text):
	"""
	Create mapping of unique chars to integers
	returns unique chars, index
	"""
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))

	return chars, char_to_int

def encode_sequence():

	raw_text = get_raw_text(filename)
	chars, char_to_int = get_char_to_int(raw_text)

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

	# indicate model depth
	model_size=input('Small(S) OR Large(L): ')

	# define the LSTM model
	model = Sequential()

	# parameters
	NUM_HIDDEN_UNITS = 256
	ACTIVATION = 'softmax'
	LOSS = 'categorical_crossentropy'
	OPTMIZER = 'adam'
	NUM_EPOCHS = 30
	DROPOUT=0.2


	print(X.shape, y.shape)

	# Dropout 20%

	if model_size.lower() == 's':
		model.add(LSTM(NUM_HIDDEN_UNITS, input_shape=(X.shape[1], X.shape[2])))
		model.add(Dropout(DROPOUT))
		model.add(Dense(y.shape[1], activation=ACTIVATION))
		model.compile(loss=LOSS, optimizer=OPTMIZER)

		# define the checkpoint 
		filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]

		# fit the model
		model.fit(X, y, epochs=NUM_EPOCHS, batch_size=128, callbacks=callbacks_list)

	else:
		model.add(LSTM(NUM_HIDDEN_UNITS, input_shape=(
			X.shape[1], X.shape[2]), return_sequences=True))
		model.add(Dropout(DROPOUT))
		model.add(LSTM(NUM_HIDDEN_UNITS))
		model.add(Dropout(DROPOUT))
		model.add(Dense(y.shape[1], activation=ACTIVATION))
		model.compile(loss=LOSS, optimizer=OPTMIZER)

		# define the checkpoint
		filepath="weights-ulysses-{epoch:02d}-{loss:.4f}-512.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]

		# fit the model
		model.fit(X, y, epochs=NUM_EPOCHS, batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':

	alice="input/"+'alice-in-wonderland.txt'
	colombiano="input/"+'colombiano.txt'
	ulysses = "input/ulysses.txt"

	filename = ulysses

	encode_sequence()