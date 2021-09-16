"""MinREM Fix"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.postprocess import default_map

def WakeToREM(Y_hat, p, map=default_map):
	"""
	Wake to REM addon. Brute repair of invalid
	transition from waking to REM sleep.

	Fix is conducted by ablating the entire REM episode proceeding an
	invalid AW/QW to R transition. The post-processed prediction is
	the highest probability of the remaining 3 valid states.

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

	Returns
	-------
	p : array-like, shape=(n_samples,)
		The post-processed predictions.
	"""
	for i in range(len(p)-1):
		if p[i] < map['NR'] and p[i+1] == map['R']:
			j = i+1
			while j < len(p) and p[j] == map['R']:
				j += 1
			p[i+1:j] = np.argmax(Y_hat[i+1:j,:map['R']], axis=1)
	return p
