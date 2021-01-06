"""Merge Processor"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.io import Dataset
from sleepens.utils import aggregate

def merge(ds, name=None, feature='MERGE', method='mean', axis=1):
	"""
	Merge features into a single feature
	by the given method.

	Parameters
	----------
	features : list
		List of 1-D feature ndarrays.

	method : {'mean', 'median', 'max', 'min', 'sum'}, default='mean'
		Method to compute the summary statistic of the band.

	Returns
	-------
	feature : ndarray
		Merged feature data.
	"""
	data = ds.data
	data = aggregate(data, method, axis=axis)
	data = data.reshape(-1,1)
	return Dataset(name=name, features=[feature], data=data)
