"""Check Functions"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

def is_float(string):
	"""
	Check if `string` can be converted
	into a float.

	Parameters
	----------
	string : str
		String to check.

	Returns
	-------
	check : bool
		True if `string` can be converted.
	"""
	try:
		float(string)
		return True
	except ValueError:
		return False

def is_int(string):
	"""
	Check if `string` can be converted
	into an int.

	Parameters
	----------
	string : str
		String to check.

	Returns
	-------
	check : bool
		True if `string` can be converted.
	"""
	try:
		int(string)
		return True
	except ValueError:
		return False

def check_XY(X=None, Y=None):
	"""
	Check if `X` and/or `Y` have the same length
	returning well-formatted ndarrays.
	Note this does not guarantee `X` and `Y`
	will work with any models as it does not rigorously
	inspect data types, only that it is in tabular format.

	Parameters
	----------
	X : array-like, default=None
		Array `X`.

	Y : array-like, default=None
		Array `Y`.

	Returns
	-------
	X : ndarray
		Returns validated `X`.
		If not provided, returns None.

	Y : ndarray
		Returns validated `Y`.
		If not provided, returns None.

	If both `X` and `Y` are provided or neither are
	provided, the results will be returned as a tuple
	in order: (X, Y).
	"""
	if X is not None and Y is not None:
		X, Y = check_X(X), check_Y(Y)
		if X.shape[0] != Y.shape[0]:
			raise ValueError("X and Y do not have the same length")
		return X, Y
	elif X is not None:
		return check_X(X), None
	elif Y is not None:
		return None, check_Y(Y)
	else:
		return None, None

def check_X(X):
	"""
	Convert `X` into an ndarray
	and allow conversion to a 2-dimensional array
	if currently 1-dimensional.

	Parameters
	----------
	X : array-like
		Array `X`.

	Returns
	-------
	X : ndarray
		Returns validated `X`.
	"""
	X = np.array(X)
	if X.shape[0] == X.size:
		X = X.reshape(-1,1)
	elif X.ndim < 2:
		raise ValueError("Array should be at least 2 dimensions")
	return X

def check_Y(Y):
	"""
	Convert `Y` into an ndarray.

	Parameters
	----------
	Y : array-like
		Array `Y`.

	Returns
	-------
	Y : ndarray
		Returns validated `Y`.
	"""
	Y = np.squeeze(np.array(Y))
	if Y.ndim == 0 and Y.size != 0:
		Y = Y.reshape(-1)
	if Y.shape[0] == Y.size:
		Y = Y.reshape(-1)
	if Y.ndim != 1 and Y.ndim != 2:
		raise ValueError("Array should be 1-dimensional or 2-dimensional")
	return Y
