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

	data = open(_file, 'r').read()
	data = data.lower()
	chars = list(set(data))
	data_size, vocab_size = len(data), len(chars)

	print("data has {} characters, {} unique.".format(data_size, vocab_size))

	char_to_ix = { ch : i for i, ch in enumerate(sorted(chars))}
	ix_to_char = { i : ch for i, ch in enumerate (sorted(chars))}

	return char_to_ix, ix_to_char

def rnn_step_forward(x_t, a_prev, params):
	""" Single forward step for RNN """
	Wax = params['Wax']
	Waa = params['Waa']
	Wya = params['Wya']
	ba = params['ba']
	by = params['by']

	a_next 	= np.tanh( np.dot(Waa, a_prev) + np.dot(Wax, x_t) + ba)
	y_t		= softmax( np.dot(Wya, a_next) + by)

	cache = (a_next, y_t, x_t, params)

	return a_next, y_t, x_t, params

def rnn_forward(x, a0, params):
	""" Forward propagation of RNN """

	caches = list()

	Nx, Tx, m = x.shape # 

	for t in range(Tx):
		# 



### Helper functions

def softmax(x):
	"""Compute the softmax function for each row of the input x.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.

    Return:
    x -- after application of the softmax function
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # x is a matrix
        col_max = np.max(x, axis=1)                # find max across columns
        x -= np.reshape(col_max, (x.shape[0],1))   # need to reshape matching the initial row size
        num = np.exp(x)
        denum = np.reshape(np.sum(np.exp(x), axis=1), (x.shape[0], 1)) # need to reshape before division
        x = np.divide(num, denum)
    else:
        # x is a vector
        x -= np.max(x)
        x = np.divide(np.exp(x), np.sum(np.exp(x)))

    assert x.shape == orig_shape
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))