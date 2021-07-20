import numpy as np

from sleepens.addon._base_addon import AbstractAddon

class WakeToREM(AbstractAddon):
	"""
	Wake to REM addon. Brute repair of invalid
	transition from waking to REM sleep.

	Fix is conducted by ablating the entire REM episode proceeding an
	invalid AW/QW to R transition. The post-processed prediction is
	the highest probability of the remaining 3 valid states.

	Parameters
	----------
	verbose : int, default=0
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.
	"""
	def __init__(self, verbose=0):
		AbstractAddon.__init__(self, verbose=verbose)

	def addon(self, Y_hat, p):
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

		Returns
		-------
		p : array-like, shape=(n_samples,)
			The post-processed predictions.
		"""
		for i in range(len(p)-1):
			if p[i] <= 1 and p[i+1] == 3:
				j = i+1
				while j < len(p) and p[j] == 3:
					j += 1
				p[i+1:j] = np.argmax(Y_hat[i+1:j,:3], axis=1)
		return p
