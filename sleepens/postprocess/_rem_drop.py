"""REMDrop Fix"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.postprocess import default_map

def REMDrop(Y_hat, p, map=default_map, window=3, threshold=0.1):
	"""
	REM Drop Addon. Drop the rest of a REM episode
	if the probability declines past a moving average.

	Fix is conducted by determining if the probability of REM becomes
	lower than a moving average and dropping all subsequent REM states
	in favour of the next highest probable state.

	Parameters
	----------
	Y_hat : array-like, shape=(n_samples, n_classes)
		The raw prediction probabilities.

	p : array-like, shape=(n_samples,)
		The predictions to process. If no Addon
		processing was done prior to this, `p`
		corresponds with Y_hat.

	map : dict
		Mapping the label values to some
		set of integers.

	window : int, default=3
		Running average window size.

	threshold : float, default=0.1
		Maximum allowable drop in REM probability to
		maintain REM episode.

	Returns
	-------
	p : array-like, shape=(n_samples,)
		The post-processed predictions.
	"""
	for i in range(len(p)):
		if p[i] == map['R']:
			end = i
			while end < len(p) and p[end] == 3:
				end += 1
			if end - i > window:
				j, k = end-window, end
				post_avg, avg = 0, 0
				while k - i > window and avg - post_avg <= threshold:
					post_avg = np.mean(Y_hat[j:k,map['R']])
					j, k = j-1, k-1
					avg = np.mean(Y_hat[j:k,map['R']])
				if avg - post_avg > threshold:
					p[k:end] = np.argmax(Y_hat[k:end,:map['R']], axis=1)
			i = end
	return p
