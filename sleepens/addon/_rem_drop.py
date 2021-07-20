import numpy as np

from sleepens.addon._base_addon import AbstractAddon

class REMDrop(AbstractAddon):
	"""
	REM Drop Addon. Drop the rest of a REM episode
	if the probability declines past a moving average.

	Fix is conducted by determining if the probability of REM becomes
	lower than a moving average and dropping all subsequent REM states
	in favour of the next highest probable state.

	Parameters
	----------
	verbose : int, default=0
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.
	"""
	def __init__(self, verbose=0):
		AbstractAddon.__init__(self, verbose=verbose)

	def addon(self, Y_hat, p, window=3, threshold=0.1):
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
			if p[i] == 3:
				end = i
				while end < len(p) and p[end] == 3:
					end += 1
				if end - i > window:
					j, k = end-window, end
					post_avg, avg = 0, 0
					while k - i > window and avg - post_avg <= threshold:
						post_avg = np.mean(Y_hat[j:k,3])
						j, k = j-1, k-1
						avg = np.mean(Y_hat[j:k,3])
					if avg - post_avg > threshold:
						p[k:end] = np.argmax(Y_hat[k:end,:3], axis=1)
				i = end
		return p
