"""MinREM Fix"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.postprocess import default_map

def MinREM(Y_hat, p, map=default_map, min_rem=3):
	"""
	Minimum REM length addon. Force REM episodes
	to be of minimum length.

	Fix is conducted by a backwards pass through predictions
	and brute repairing predictions to satisfy the minimum
	episode length requirement.

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

	min_rem : int, default=3
		Minimum length of a REM episode

	Returns
	-------
	p : array-like, shape=(n_samples,)
		The post-processed predictions.
	"""
	in_rem = 0
	for i in range(len(p)-1,-1,-1):
		if p[i] == map['R'] : in_rem += 1
		elif in_rem > 0 and in_rem < min_rem:
			p[i] = map['R']
			in_rem += 1
		else : in_rem = 0
	return p
