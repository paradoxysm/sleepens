import numpy as np

from sleepens.addon._base_addon import AbstractAddon

class REMSensitivity(AbstractAddon):
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
	verbose : int, default=0
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.
	"""
	def __init__(self, verbose=0):
		AbstractAddon.__init__(self, verbose=verbose)

	def addon(self, Y_hat, p, avg_threshold=0.08, init_threshold=0.03, min_threshold=0.02,
					window=2, min_size=3):
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
			if p[i] == 3 : rem_count += 1
			if rem_count == 0 and p[i+1] == 2 and Y_hat[i,3] >= init_threshold:
				avg = np.mean(Y_hat[i:i+window,3])
				if avg >= avg_threshold : rem_count += 1
				else : rem_count = 0
			if rem_count > 0 and p[i] != 0 and p[i] != 3 and Y_hat[i,3] >= min_threshold:
				avg = np.mean(Y_hat[i:i+window,3])
				if avg >= avg_threshold:
					rem_count += 1
					if rem_count > min_size : p[i] = 3
				else : rem_count = 0
			elif p[i] != 3 : rem_count = 0
			if rem_count == min_size : p[i:i+min_size] = 3
			i -= 1
		return p
