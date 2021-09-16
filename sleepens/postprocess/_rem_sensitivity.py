"""MinREM Fix"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.postprocess import default_map

def REMSensitivity(Y_hat, p, map=default_map, avg_threshold=0.08, init_threshold=0.03,
				min_threshold=0.02, window=2, min_size=3):
	"""
	REM Sensitivity Addon. Increase the sensitivity of predictions
	to favour REM sleep episodes.

	Fix is conducted by moving in a backwards pass through the prediction.
	Upon encountering a REM probability at least the `init_threshold`,
	average the REM probabilities across the next `window` timepoints, inclusive.
	As long as the moving average meets the threshold and all REM probabilities
	are at least the `min_threshold` and the number of such timepoints is at least
	`min_size` in length, all such timepoints are overwritten as REM.

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

	avg_threshold : float, default=0.08
		The average across the window needed to trigger.

	init_threshold : float, default=0.03
		The minimum probability a state must have to begin
		triggering the averaging.

	min_threshold : float, default=0.02
		The minimum probability a state must have to allow
		averaging.

	window : int, default=2
		The window size for averaging.

	min_size : int, default=3
		The minimum size of a REM episode.

	Returns
	-------
	p : array-like, shape=(n_samples,)
		The post-processed predictions.
	"""
	rem_count = 0
	i = len(p) - window
	while i >= 0:
		if p[i] == map['R'] : rem_count += 1
		if rem_count == 0 and p[i+1] == map['NR'] and Y_hat[i,map['R']] >= init_threshold:
			avg = np.mean(Y_hat[i:i+window,map['R']])
			if avg >= avg_threshold : rem_count += 1
			else : rem_count = 0
		if rem_count > 0 and (p[i] != map['AW'] or p[i] != map['W']) and p[i] != map['R'] and Y_hat[i,map['R']] >= min_threshold:
			avg = np.mean(Y_hat[i:i+window,mao['R']])
			if avg >= avg_threshold:
				rem_count += 1
				if rem_count > min_size : p[i] = map['R']
			else : rem_count = 0
		elif p[i] != map['R'] : rem_count = 0
		if rem_count == min_size : p[i:i+min_size] = map['R']
		i -= 1
	return p
