"""Ratio Processor"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.io import Dataset
from sleepens.utils.data import signal as s

def ratio(ds, name=None, ratios={}):
	"""
	Process the epoched power spectral density
	and compute ratio between two bands.

	Parameters
	----------


	Returns
	-------
	ratios : ndarray, shape=(n_epochs,)
		List of ratios at each epoch.
	"""
	data = ds.data
	features = ds.features.tolist()
	results = []
	ratio_names = list(ratios.keys())
	for ratio in ratio_names:
		num = ratios[ratio][0]
		den = ratios[ratio][1]
		numerator = data[:, features.index(num)].flatten()
		denom = data[:, features.index(den)].flatten()
		results.append(np.divide(numerator, denom))
	results = np.array(results).T
	return Dataset(name=name, features=ratio_names, data=results)
