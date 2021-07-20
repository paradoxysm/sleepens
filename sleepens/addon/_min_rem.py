import numpy as np

from sleepens.addon._base_addon import AbstractAddon

class MinREM(AbstractAddon):
	"""
	Minimum REM length addon. Force REM episodes
	to be of minimum length.

	Fix is conducted by a backwards pass through predictions
	and brute repairing predictions to satisfy the minimum
	episode length requirement.

	Parameters
	----------
	verbose : int, default=0
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.
	"""
	def __init__(self, verbose=0):
		AbstractAddon.__init__(self, verbose=verbose)

	def addon(self, Y_hat, p, min_rem=3):
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

		min_rem : int, default=3
			Minimum length of a REM episode

		Returns
		-------
		p : array-like, shape=(n_samples,)
			The post-processed predictions.
		"""
		in_rem = 0
		for i in range(len(p)-1,-1,-1):
			if p[i] == 3 : in_rem += 1
			elif in_rem > 0 and in_rem < min_rem:
				p[i] = 3
				in_rem += 1
			else : in_rem = 0
		return p
