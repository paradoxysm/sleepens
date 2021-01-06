"""Normalization Functions"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.utils.data.transform import log_transform

def demean(data, axis=0):
	"""
	Demean the data by subtracting the mean from
	all samples.

	Parameters
	----------
	data : ndarray
		Data to be normalized.

	axis : int, default=0
		Axis to demean along.

	Returns
	-------
	demeaned : ndarray
		Demeaned data.
	"""
	return data - np.mean(data, axis=axis)

def normalize(data, mu=0, sigma=1, axis=0):
	"""
	Normalize `data` such that the mean is
	`mu` and standard deviation is `sigma`.

	Parameters
	----------
	data : ndarray
		Data to be normalized.

	mu : float, default=0
		Mean of the normalized data.

	sigma : float, default=1
		Standard deviation of the normalized data.

	axis : int, default=0
		Axis to normalize along.

	Returns
	-------
	norm : ndarray
		Normalized data.
	"""
	mean = np.mean(data, axis=axis) - mu
	std = np.std(data, axis=axis) * sigma
	return (data - mean) / std

def max_normalize(data, scale=(0,1), axis=0):
	"""
	Normalize `data` into an interval defined by `scale`.

	Parameters
	----------
	data : ndarray
		Data to be normalized.

	scale : tuple, shape=(2,), default=(0,1)
		Normalization interval. All data will be bounded
		within this interval such that the maximum of `data`
		is the maximum of `scale`.

	axis : int, default=0
		Axis to normalize along.

	Returns
	-------
	norm : ndarray
		Normalized data.
	"""
	factor = (scale[1] - scale[0]) / (data.max(axis=axis) - data.min(axis=axis))
	return factor * (data - data.min(axis=axis)) + scale[0]

def prob_normalize(data, scale=1, axis=0):
	"""
	Normalize `data` as a probability.
	Calculated as x divided by the sum of
	data values for each sample in the data.

	Parameters
	----------
	data : ndarray
		Data to be normalized.

	scale : scalar, default=1
		Normalization scale. A probability of 1
		corresponds to `scale`.

	axis : int, default=0
		Axis to normalize along.

	Returns
	-------
	norm : ndarray
		Normalized data.
	"""
	return (data / np.sum(data, axis=axis)) * scale

def log_normalize(data, base=np.e, scale=(0,1), axis=0):
	"""
	Log normalize `data`.

	Parameters
	----------
	data : ndarray
		Data to be normalized.

	base : {scalar, 'mean', 'median'}, default=np.e
		Log base for normalization. Must be one of:
		 - scalar : Constant base.
		 - 'mean' : Set the base to the mean of the data.
		 - 'median' : Set the base to the median of the data.

	scale : tuple, shape=(2,), default=(0,1)
		Normalization interval. All data will be bounded
		within this interval such that the maximum of `data`
		is the maximum of `scale`.

	axis : int, default=0
		Axis to normalize along.

	Returns
	-------
	norm : ndarray
		Normalized data.
	"""
	data = max_normalize(data, scale=(1,2), axis=axis)
	return max_normalize(log_transform(data, base=base, axis=axis),
							scale=scale, axis=axis)
