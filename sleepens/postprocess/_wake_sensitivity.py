"""WakeSensitivity Fix"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.postprocess import default_map

def WakeSensitivity(Y_hat, p, map=default_map, threshold=0.5):
	"""
	Wake Sensitivity Addon. Increase the sensitivity
	of waking due to split of active and quiet waking periods.

	Fix is conducted by determining when the combined probability
	of the AW and QW states for a given timepoint sum to above the given
	threshold. When this condition is satisified, the prediction is overwritten
	to the waking state with the higher probability.

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

	threshold : float, [0,1), default=0.5
		The threshold for AW, QW states to reclassify
		as waking.

	Returns
	-------
	p : array-like, shape=(n_samples,)
		The post-processed predictions.
	"""
	for i in range(len(p)):
		if np.sum(Y_hat[i,:map['NR']]) >= threshold and p[i] >= map['NR']:
			p[i] = np.argmax(Y_hat[i,:map['NR']])
	return p
