"""Transition Fix"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from copy import deepcopy

from sleepens.addon._base_addon import AbstractAddon

class TransitionFix(AbstractAddon):
	"""
	Transition Fix Addon. Ensure predictions follow
	physiologically feasible state transitions.

	Fix is conducted when an invalid transition is discovered.
	Half a forward pass is conducted from the invalid transition
	point, adjusting state predictions until the transition error resolves.
	In the forward pass, predictions at time `t` are considered immutable while
	`t+1` can be fixed.

	Subsequently, a backwards pass from the end of the forward pass is done,
	adjusting state predictions until the transition error resolves.
	In the backwards pass, predictions at `t+1` are considered immutable while
	`t` can be fixed.

	Finally, the remaining half of the forward pass is conducted until reaching
	the point of the original error. The pass with the highest internal probability,
	as determined by the transition matrix, is used to determine the new
	state predictions.

	It is recommended this be used as the final Addon to guarantee
	state transitions.

	Parameters
	----------
	verbose : int, default=0
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.
	"""
	def __init__(self, transition_matrix=None, verbose=0):
		AbstractAddon.__init__(self, verbose=verbose)

	def addon(self, Y_hat, p, transition_matrix=None):
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

		transition_matrix : ndarray, None, shape=(n_classes, n_classes)
			The matrix of possible transitions. Rows represent
			timepoint `t` while columns represent `t+1`.
			0 represents an invalid transition, while non-zero numbers
			act as a multiplicative factor of the probability to
			determine the most likely transition fix sequence.

		Returns
		-------
		p : array-like, shape=(n_samples,)
			The post-processed predictions.
		"""
		if transition_matrix is None:
			transitions = np.array([	[1, 1, 1, 0],
											[1, 1, 1, 0],
											[1, 1, 1, 1],
											[1, 1, 0, 2]])
		else : transitions = transition_matrix

		for i in range(len(p)-1):
			if transitions[p[i], p[i+1]] == 0:
				k = i
				p_forward, p_f = 0, deepcopy(p)
				while k < len(p)-1 and transitions[p_f[k], p_f[k+1]] == 0:
					p_forward += np.max(Y_hat[k+1]*transitions[p_f[k]])
					p_f[k+1] = np.argmax(Y_hat[k+1]*transitions[p_f[k]])
					k += 1
				k += 1
				end = k
				inverse = transitions.T
				p_back, p_b = 0, deepcopy(p)
				if k == len(p):
					p_back += np.max(Y_hat[k-1])
					p_b[k-1] = np.argmax(Y_hat[k-1])
					k -= 1
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
