"""Calculate Functions"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

def calculate_batch(batch_size, length):
	"""
	Calculate the batch size for the data of given length.

	Parameters
	----------
	batch_size : int, float, default=None
		Batch size for training. Must be one of:
		 - int : Use `batch_size`.
		 - float : Use `batch_size * n_samples`.
		 - None : Use `n_samples`.

	length : int
		Length of the data to be batched.

	Returns
	-------
	batch : int
		Actual batch size.
	"""
	if batch_size is None : return length
	elif isinstance(batch_size, int) and batch_size > 0 and \
			batch_size <= length:
		return batch_size
	elif isinstance(batch_size, float) and 0 < batch_size <= 1:
		return int(batch_size * length)
	else:
		raise ValueError("Batch size must be None, an int less than %d," % length,
							"or a float within (0,1]")

def calculate_weight(Y, n_classes, class_weight=None, weights=None):
	"""
	Calculate the weights applied to the predicted labels,
	combining class weights and sample weights.

	Parameters
	----------
	Y : array-like, shape=(n_samples,)
		Target labels as integers.

	n_classes : int
		Number of classes.

	class_weight : dict, 'balanced', or None, default=None
		Weights associated with classes in the form
		`{class_label: weight}`. Must be one of:
		 - None : All classes have a weight of one.
		 - 'balanced': Class weights are automatically calculated as
						`n_samples / (n_samples * np.bincount(Y))`.

	weights : array-like, shape=(n_samples,), default=None
		Sample weights. If None, then samples are equally weighted.

	Returns
	-------
	weights : array-like, shape=(n_samples,)
		Weights combining sample weights and class weights.
	"""
	if class_weight is None and weights is None : return np.ones(len(Y))
	elif weights is None : weights = np.ones(len(Y))
	d = class_weight
	if isinstance(d, str) and d == 'balanced':
		l = len(Y) / (n_classes * np.bincount(Y))
		d = {k: l[k] for k in range(len(l))}
	if isinstance(d, dict):
		class_weights = np.array([d[k] for k in Y])
	elif d is None : class_weights = np.ones(len(Y))
	else : raise ValueError("Class Weight must either be a dict or 'balanced' or None")
	return weights * class_weights

def calculate_bootstrap(bootstrap_size, length):
	"""
	Calculate the bootstrap size for the data of given length.

	Parameters
	----------
	bootstrap_size : int, float, default=None
		Bootstrap size for training. Must be one of:
		 - int : Use `bootstrap_size`.
		 - float : Use `bootstrap_size * n_samples`.
		 - None : Use `n_samples`.

	length : int
		Length of the data to be bootstrapped.

	Returns
	-------
	bootstrap : int
		Actual bootstrap size.
	"""
	if bootstrap_size is None:
		return length
	elif isinstance(bootstrap_size, int) and bootstrap_size > 0:
		return bootstrap_size
	elif isinstance(bootstrap_size, float) and 0 < bootstrap_size <= 1:
		return int(bootstrap_size * length)
	else : raise ValueError("Bootstrap Size must be None, a positive int or float in (0,1]")
