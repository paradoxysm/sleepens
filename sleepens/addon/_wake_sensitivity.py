import numpy as np

from sleepens.addon._base_addon import AbstractAddon

class RemFix(AbstractAddon):
	def __init__(self, verbose=0):
		AbstractAddon.__init__(self, verbose=verbose)

	def addon(self, Y_hat, p, threshold=0.5):
		for i in range(len(p)):
			if np.sum(Y_hat[i,:2]) >= threshold and p[i] > 1:
				p[i] = np.argmax(Y_hat[i,:2])
		return p
