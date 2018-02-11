import numpy as np

class RNN(object):
	"""docstring for RNN"""
	def __init__(self, arg):
		super(RNN, self).__init__()
		self.arg = arg
		
	def forward_step(self, x):
		self.h	= np.tanh( np.dot(self.W_hh, self.h) + np.dot(W_hx, x) + b )
		self.y	= np.dot( self.W_yh, self.h )