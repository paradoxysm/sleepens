import numpy as np

from sleepens.addon._base_addon import AbstractAddon

class RemFix(AbstractAddon):
	def __init__(self, verbose=0):
		AbstractAddon.__init__(self, verbose=verbose)

	def addon(self, Y_hat, p, avg_threshold=0.08, init_threshold=0.03, min_threshold=0.02,
					window=2, min_size=3):
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
