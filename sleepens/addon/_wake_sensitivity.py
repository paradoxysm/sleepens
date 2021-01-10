import numpy as np

from sleepens.addon._base_addon import AbstractAddon

class WakeSensitivity(AbstractAddon):
	"""
	Wake Sensitivity Addon. Increase the sensitivity
	of waking due to split of active and quiet waking periods.

	Fix is conducted by determining when the combined probability
	of the AW and QW states for a given timepoint sum to above the given
	threshold. When this condition is satisified, the prediction is overwritten
	to the waking state with the higher probability.

	Parameters
	----------
	verbose : int, default=0
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.
	"""
	def __init__(self, verbose=0):
		AbstractAddon.__init__(self, verbose=verbose)

	def addon(self, Y_hat, p, threshold=0.5):
		"""
		Post-process predictions.

		Parameters
		----------
		Y_hat : array-like, shape=(n_samples, n_classes)
			The raw prediction probabilities.

		p : array-like, shape=(n_samples,)
			The predictions to process. If no Addon
			processing was done prior to this, `p`
			corresponds with Y_hat.

		threshold : float, [0,1), default=0.5
			The threshold for AW, QW states to reclassify
			as waking.

		Returns
		-------
		p : array-like, shape=(n_samples,)
			The post-processed predictions.
		"""
		for i in range(len(p)):
			if np.sum(Y_hat[i,:2]) >= threshold and p[i] > 1:
				p[i] = np.argmax(Y_hat[i,:2])
		return p
