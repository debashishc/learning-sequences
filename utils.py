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