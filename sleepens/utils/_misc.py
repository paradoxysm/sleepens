"""Miscellaneous Functions"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

def aggregate(data, method, axis=0):
	"""
	Summarize data based on the `method`.

	Parameters
	----------
	data : ndarray
		Data.

	method : {'mean', 'median', 'min', 'max', 'sum', 'count'}
		Method to summarize data.
		Must be one of:
		 - 'mean' : Mean of variances above threshold.
		 - 'median' : Median of variances above threshold.
		 - 'max' : Max of variances above threshold.
		 - 'min' : Min of variances above threshold.
		 - 'sum' : Sum of variances above threshold.
		 - 'count' : Count of variances above threshold.

	axis : int, default=0
		Axis to summarize along.

	Returns
	-------
	summarized : float
		The summarized data.
	"""
	if method == 'mean' : return np.mean(data, axis=axis)
	elif method == 'median' : return np.median(data, axis=axis)
	elif method == 'min' : return np.min(data, axis=axis)
	elif method == 'max' : return np.max(data, axis=axis)
	elif method == 'sum' : return np.sum(data, axis=axis)
	elif method == 'count': return data.shape[axis]
	else: raise ValueError("Method must be 'mean','median','min','max','sum','count'")

def determine_threshold(data, threshold, axis=0):
	"""
	Determine the real value threshold of the `data`
	given a specified `threshold` type.

	Parameters
	----------
	data : ndarray
		Data.

	threshold : {'mean', 'median', 'min', 'max', 'sum', 'count'}
		Method to determine threshold.
		Must be one of:
		 - 'mean' : Mean.
		 - 'median' : Median.
		 - '%<k>' : k-th percentile where <k> is an int.

	axis : int, default=0
		Axis to calculate along.

	Returns
	-------
	threshold : float or ndarray
		The threshold or list of thresholds.
	"""
	if isinstance(threshold, (float, int)) : pass
	elif threshold == 'mean': threshold = np.mean(data, axis=axis)
	elif threshold == 'median': threshold = np.median(data, axis=axis)
	elif threshold[0] == '%' and threshold[1:].isdigit():
		threshold = np.percentile(data, int(threshold[1:]), axis=axis)
	else : raise ValueError("Threshold must be 'mean','median','%<k>',float/int")
	return threshold
