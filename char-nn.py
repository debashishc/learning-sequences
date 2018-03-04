"""
Replicating results from Andrej Karpathy's blog using RNN, LSTM and possibly GRU:
https://karpathy.github.io/2015/05/21/rnn-effectiveness/

"""

import numpy as np
import random

input_file = 'input.txt'

# hyperparameters
hidden_size = 100
seq_len = 25
learning_rate = 0.001

def get_file_info(_file):
	"""
	Given a input text file, retrieve two dictionaries: one mapping each character to an index (0-26)
	and another mapping each index to its corresponding character.
	"""

	with open(_file, 'r', encoding='utf-8') as f:
		data = f.read().lower()

	chars = sorted(list(set(data)))
	data_size, vocab_size = len(data), len(chars)

	print("data has {} characters, {} unique.".format(data_size, vocab_size))

	char_indices = dict(ch : i for i, ch in enumerate(sorted(chars)))
	indices_char = dict(i : ch for i, ch in enumerate (sorted(chars)))

	return char_indices, indices_char

def rnn_step_forward(params, x_t, a_prev):
	""" Single forward step for RNN """
	Wax = params['Wax']
	Waa = params['Waa']
	Wya = params['Wya']
	ba = params['ba']
	by = params['by']

	a_next 	= np.tanh( np.dot(Waa, a_prev) + np.dot(Wax, x_t) + ba)	# hidden state
	y_t		= softmax( np.dot(Wya, a_next) + by)					# next char probabilities

	cache = (a_next, y_t, x_t, params)

	return a_next, y_t, x_t, params

def rnn_forward(a0, X, Y, params, vocab_size):
	""" Forward propagation of RNN 
	
	Finding loss and get a cache for back-propagation
	"""

	# initiate the 
	a, x, y_pred = {}, {}, {}

	a[-1] = np.copy(a0)

	loss = 0

	for t in range(len(X)):

		x[t] = np.zeroes((vocab_size, 1))

		#
		if X[t] != None:
			x[t][X[t]] = 1

			a[t], y_pred[t], _ , _ = rnn_step_forward(params, x[t], a[t-1])

			loss -= np.log(y_pred[t][Y[t],0])

		cache = (x, a, y_pred)

	return loss, cache