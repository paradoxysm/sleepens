"""Smoothing Functions"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

def moving_average(data, window, weight=False, axis=0):
	"""
	Compute a moving average of the data.
	Moving average is calculated as average
	of the last `window` samples.

	Parameters
	----------
	data : ndarray
		Data.

	window : int
		Size of the moving average window.

	weight : bool, default=False
		Compute a weighted moving average.

	axis : int, default=0
		Axis to smooth along.

	Returns
	-------
	ma : ndarray
		Moving average of the data.
	"""
	result = []
	for i in range(len(data)):
		if i - window < 0:
			start = 0
		else:
			start = i - window + 1
		if weight:
			weights = np.arange(i-start+1) + 1
			sum = np.sum(data[start:i+1] * weights, axis=axis)
			avg = sum / np.sum(weights)
		else:
			sum = np.sum(data[start:i+1], axis=axis)
			avg = sum / window
		result.append(avg)
	return np.array(result)

def hull_moving_average(data, window, weight=False, axis=0):
	"""
	Compute a Hull moving average of the data.
	This is less susceptible to lag effects.
	Calculated as the moving average of
	Moving average is calculated as average
	of the last `window` samples.

	Parameters
	----------
	data : ndarray
		Data.

	window : int
		Size of the moving average window.

	weight : bool, default=False
		Compute a weighted moving average.

	axis : int, default=0
		Axis to smooth along.

	Returns
	-------
	hma : ndarray
		Hull moving average of the data.
	"""
	period = int(np.sqrt(window))
	half = int(window / 2)
	if half == 0 : half = 1
	halfaverage = moving_average(data, half, weight, axis=axis)
	fullaverage = moving_average(data, window, weight, axis=axis)
	intermediate = 2 * halfaverage - fullaverage
	result = moving_average(intermediate, period, weight, axis=axis)
	return result
