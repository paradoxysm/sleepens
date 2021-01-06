import numpy as np
from copy import deepcopy

from sleepens.addon._base_addon import AbstractAddon

class RemFix(AbstractAddon):
	def __init__(self, verbose=0):
		AbstractAddon.__init__(self, verbose=verbose)

	def addon(self, Y_hat, p):
		transitions = np.array([	[1, 1, 1, 0],
									[1, 1, 1, 0],
									[1, 1, 1, 1],
									[1, 1, 0, 2]])
		for i in range(len(p)-1):
			if transitions[p[i], p[i+1]] == 0:
				k = i
				p_forward, p_f = 0, deepcopy(p)
				while k < len(p) and transitions[p_f[k], p_f[k+1]] == 0:
					p_forward += np.max(Y_hat[k+1]*transitions[p_f[k]])
					p_f[k+1] = np.argmax(Y_hat[k+1]*transitions[p_f[k]])
					k += 1
				k += 1
				end = k
				inverse = transitions.T
				p_back, p_b = 0, deepcopy(p)
				while k > 0 and (k >= i or inverse[p_b[k], p_b[k-1]] == 0):
					p_back += np.max(Y_hat[k-1]*inverse[p_b[k]])
					p_b[k-1] = np.argmax(Y_hat[k-1]*inverse[p_b[k]])
					k -= 1
				start = k
				k -= 1
				while k < i:
					p_forward += np.max(Y_hat[k+1]*transitions[p_f[k]])
					p_f[k+1] = np.argmax(Y_hat[k+1]*transitions[p_f[k]])
					k += 1
				if p_forward >= p_back : p = p_f
				else : p = p_b
		return p
