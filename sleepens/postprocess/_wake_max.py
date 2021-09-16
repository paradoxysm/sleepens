"""WakeMax Fix"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.postprocess import default_map

def WakeMax(Y_hat, p, map=default_map):
	"""
	Wake Max Addon. Increase the sensitivity
	of waking due to split of active and quiet waking periods,
	thresholded by NR state probability.

	Fix is conducted by determining when the combined probability
	of the AW and QW states for a given timepoint sum to above the probability
	of the NR state for the given timepoint. When this condition is satisified,
	the prediction is overwritten to the waking state with the higher probability.

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
	for i in range(len(p)):
		if p[i] == map['NR'] and np.sum(Y_hat[i,:map['NR']]) > Y_hat[i,map['NR']]:
			p[i] = np.argmax(Y_hat[i,:map['NR']])
	return p
