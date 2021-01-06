"""Data Manipulation Functions"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.utils import check_X

def time_array(X, n):
	"""
	Sequence time series data with a rolling window of size `n`.

	Parameters
	----------
	X : array-like, shape=(n_samples, n_features)
		Time series data.

	n : int
		Window size.

	Returns
	-------
	ta : array-like, shape=(n_samples-n+1, n_features*n)
		Time arrayed data.
	"""
	X = check_X(X)
	if X.ndim <= 1 : X = X.reshape(-1,1)
	if len(X) < n : raise ValueError("Length of array is smaller than n")
	end = len(X) - n + 1
	ta = X[:end]
	for i in range(1,n):
		ta = np.concatenate((ta, X[i:end+i]), axis=1)
	return ta

def separate_by_label(data, labels):
	"""
	Create a dictionary grouping data by labels
	in order.

	Parameters
	----------
	data : array-like, shape=(n_samples, n_features)
		Data.

	labels : array-like, shape=(n_samples,)
		Labels for the given `data`.

	Returns
	-------
	separated : dict
		Dictionary of categorized data.
	"""
	separated = {k: [] for k in labels}
	for d, l in zip(data, labels):
		separated[l].append(d)
	return separated

def one_hot(Y, cols=None):
	"""
	Convert `Y` into a one-hot encoding.

	Parameters
	----------
	Y : ndarray, shape=(n_samples,)
		Data to encode.

	cols : int or None, default=None
		Number of columns to encode.
		Must be at least the number of unique
		values in `Y`.

	Returns
	-------
	one_hot : ndarray, shape=(n_samples, n_cols)
		One-hot encoded data.
	"""
	if Y.shape[1] == cols and (set(Y.reshape(-1)) == set([0,1]) or \
			set(Y.reshape(-1)) == set([0.,1.])):
		return Y.astype(int)
	if Y.size > len(Y) : raise ValueError("Matrix is not 1D")
	Y_ = Y.reshape(-1)
	if cols is None : cols = Y_.max() + 1
	elif cols < len(np.unique(Y)) : raise ValueError("There are more classes than cols")
	if cols > 1:
		one_hot = np.zeros((len(Y), cols), dtype=int)
		one_hot[np.arange(len(Y)), Y_] = 1
		return one_hot
	else:
		Y = np.where(Y < cols, 0, 1)
		return Y.reshape(-1, 1)

def decode(Y):
	"""
	Decode one-hot encoded data
	into a 1-dimensional array.

	Parameters
	----------
	Y : ndarray, shape=(n_samples, n_cols)
		One-hot encoded data.

	Returns
	-------
	decode : ndarray, shape=(n_samples,)
		Decoded data.
	"""
	if Y.shape[0] == Y.size:
		return np.squeeze(Y)
	elif set(Y.reshape(-1)) == set([0,1]) or \
			set(Y.reshape(-1)) == set([0.,1.]):
		return np.argmax(Y, axis=1)
	else:
		raise ValueError("Y must be one-hot encoded data with ints.")
