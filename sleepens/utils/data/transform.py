"""Transformation Functions"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from scipy import stats

def log_transform(data, base=np.e, axis=0):
	"""
	Transform the data into a logarithm.

	Parameters
	----------
	data : ndarray
		Data to be log transformed.

	base : {scalar, 'mean', 'median'}, default=np.e
		Log base for normalization. Must be one of:
		 - scalar : Constant base.
		 - 'mean' : Set the base to the mean of the data.
		 - 'median' : Set the base to the median of the data.

	axis : int, default=0
		Axis to transform along.

	Returns
	-------
	transform : ndarray
		Log transformed data.
	"""
	if base == 'mean':
		base = np.mean(data, axis=axis)
	elif base == 'median':
		base = np.median(data, axis=axis)
	return np.log(data) / np.log(base)

def boxcox_transform(data, lmbda=None):
	"""
	Transform the data by a Box-Cox power transformation.

	Parameters
	----------
	data : ndarray
		Data to be transformed.

	lmbda : scalar, default=np.e
		If lmbda is not None, do the transformation for that value.
		If lmbda is None, find the lambda that maximizes
		the log-likelihood function and return it as the
		second output argument.

	Returns
	-------
	transform : ndarray
		Transformed data.

	maxlog : float, optional
		If the lmbda parameter is None, maxlog is
		the lambda that maximizes the log-likelihood function.
	"""
	return stats.boxcox(data, lmbda=lmbda)

def power_transform(data, a):
	"""
	Transform the data into by a power.

	Parameters
	----------
	data : ndarray
		Data to be power transformed.

	a : scalar
		Exponent to use for transformation.

	Returns
	-------
	transform : ndarray
		Power transformed data.
	"""
	return np.sign(data) * np.power(np.absolute(data), a)
