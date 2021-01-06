"""Oversampling Functions"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from sklearn.mixture import GaussianMixture as GMM

from sleepens.utils import separate_by_label, create_random_state

def balance(data, labels, desired=None, balance='auto',
							seed=None, verbose=0):
	"""
	Balance a dataset through generative oversampling.

	Model a generative algorithm over each of the
	desired labelled data to sample more data from
	the resulting distributions.

	Parameters
	----------
	data : array-like, shape=(n_samples, n_features)
		Data.

	labels : array-like, shape=(n_samples,)
		Labels corresponding to data.

	desired : {None, array-like}, default=None
		List of labels to oversample.
		If None, balances all labels.

	balance : {'auto', int, dict}, default='auto'
		Determines how to balance and oversample.
		Must be one of:
		 - 'auto' : Automatically balance the dataset
		 			by oversampling `desired` labels to
					match the number of samples in the majority
					label. Recommended to set `desired` to None,
					or risk not oversampling a minority label.
		 - int : Oversample `desired` labels so that total
		 			such samples reach this value.
		 - dict : Oversample each label by the value given.

	seed : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	verbose : int, default=0
		Verbosity; higher values result in
		more verbose output.

	Returns
	-------
	data : ndarray
		All oversampled data, shuffled.

	labels : ndarray
		All corresponding labels.
	"""
	if len(data) == 0:
		if verbose : print("No data to oversample")
		return np.array([]), np.array([])
	if verbose > 0 : print("Balancing Dataset")
	data = separate_by_label(data, labels)
	separated = data
	if desired is not None:
		data_labels = list(data.keys())
		for d in desired:
			if desired not in  data_labels:
				raise ValueError("Desired label is not in the data")
		separated = {d: separated[d] for d in desired}
	if balance == 'auto' or isinstance(balance, int):
		if balance == 'auto':
			if verbose > 0 : print("Balancing Dataset. Method set to 'auto'")
			target = np.max([len(data[k]) for k in data.keys()])
		else:
			if verbose > 0 : print("Balancing Dataset up to", str(balance))
			target = balance
		n_samples = {k: target - len(separated[k]) for k in separated.keys()}
		n_samples = {k: 0 if n_samples[k] < 0 else n_samples[k] for k in separated.keys()}
	elif isinstance(balance, (dict)):
		if verbose > 0 : print("Balancing Dataset. Method set to custom")
		n_samples = balance
	else:
		raise ValueError("Balance must be {'auto', int, dict}")
	if verbose > 0 : print("Oversampling")
	return _generative_oversample(separated, n_samples, seed=seed, verbose=verbose)

def scale(data, labels, factor=1, seed=None, verbose=0):
	"""
	Oversample the dataset.

	Model a generative algorithm over each of the
	desired labelled data to sample more data from
	the resulting distributions.

	Parameters
	----------
	data : array-like, shape=(n_samples, n_features)
		Data.

	labels : array-like, shape=(n_samples,)
		Labels corresponding to data.

	factor : float, default=1
		Factor to oversample the dataset by.

	seed : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	verbose : int, default=0
		Verbosity; higher values result in
		more verbose output.

	Returns
	-------
	data : ndarray
		All oversampled data, shuffled.

	labels : ndarray
		All corresponding labels.
	"""
	if factor <= 0 : raise ValueError("Factor must be positive")
	if len(data) == 0:
		if verbose > 0 : print("No data to oversample")
		return np.array([]), np.array([])
	if verbose > 0 : print("Scaling Dataset")
	separated = separate_by_label(data, labels)
	n_samples = {k: int(factor * len(separated[k])) for k in separated.keys()}
	return _generative_oversample(separated, n_samples, seed=seed, verbose=verbose)

def sample(data, labels, sizes, seed=None, verbose=0):
	"""
	Sample from the dataset.

	Model a generative algorithm over each of the
	desired labelled data to sample more data from
	the resulting distributions.

	Parameters
	----------
	data : array-like, shape=(n_samples, n_features)
		Data.

	labels : array-like, shape=(n_samples,)
		Labels corresponding to data.

	sizes : dict
		Number of samples for each class.

	seed : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	verbose : int, default=0
		Verbosity; higher values result in
		more verbose output.

	Returns
	-------
	data : ndarray
		All oversampled data, shuffled.

	labels : ndarray
		All corresponding labels.
	"""
	if len(data) == 0:
		if verbose > 0 : print("No data to oversample")
		return np.array([]), np.array([])
	if verbose > 0 : print("Sampling Dataset")
	separated = separate_by_label(data, labels)
	return _generative_oversample(separated, sizes, seed=seed, verbose=verbose)

def _generative_oversample(data_labels, n_samples, seed=None, verbose=0):
	"""
	Generatively oversample the data.

	Parameters
	----------
	data_labels : dict
		Dictionary of data categorized by label.

	n_samples : dict
		Dictionary of the number of samples to oversample
		each label.

	seed : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	verbose : int, default=0
		Verbosity; higher values result in
		more verbose output.

	Returns
	-------
	data : ndarray
		All oversampled data, shuffled.

	labels : ndarray
		All corresponding labels.
	"""
	oversampled = {}
	for label in data_labels:
		if n_samples[label] == 0 or len(data_labels[label]) < 2 : continue
		if verbose > 0 : print("\tModelling distribution for", str(label))
		model = _fit_cluster(data_labels[label], seed=seed)
		if verbose > 0 : print("\tSampling data for", str(label))
		model.weights_ = (model.weights_ / np.sum(model.weights_)).astype(np.float64)
		oversampled[label] = model.sample(n_samples[label])[0]
	if verbose > 0 : print("Collating and shuffling")
	new_set = []
	for k in oversampled.keys():
		length = len(oversampled[k])
		labels = np.array([k]*length).reshape(length, 1)
		new_set += list(np.concatenate((oversampled[k], labels), axis=1))
	create_random_state(seed=seed).shuffle(new_set)
	new_set = np.array(new_set)
	if new_set.size == 0 : return np.array([]), np.array([]).astype(int)
	return new_set[:,:-1], new_set[:,-1].astype(int)

def _fit_cluster(data, seed=None):
	"""
	Fit a Gaussian Mixture Model to the given data.

	Parameters
	----------
	data : array-like, shape=(n_samples, n_features)
		Data.

	seed : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	Returns
	-------
	model : GaussianMixture
		The best fitted Gaussian Miture Model as determined
		by the mean of the BIC and AIC for the respective model.
	"""
	data = np.array(data)
	models = []
	abic = []
	n_components = min([len(data), 10])
	for i in range(n_components):
		if len(data) < 2 * (i+1) : continue
		m = GMM(n_components=i+1, n_init=5, random_state=seed)
		m.fit(data)
		models.append(m)
		abic.append(np.mean([m.bic(data), m.aic(data)]))
	return models[np.argmin(abic)]
