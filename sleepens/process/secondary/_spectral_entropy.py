"""Spectral Entropy Processor"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.io import Dataset
from sleepens.utils.data import signal as s

def spectral_entropy(ds, name=None):
	"""
	Process the epoched power spectral density
	and compute spectral entropy.

	Parameters
	----------
	f : ndarray
		List of arrays of frequencies for each epoch.

	Pxx : ndarray
		List of power spectral densities of `data` at
		each epoch.

	Returns
	-------
	entropies : ndarray, shape=(n_epochs,)
		List of entropies at each epoch.
	"""
	f, Pxx = ds.features, ds.data
	entropies = []
	for e in range(min([len(f), len(Pxx)])):
		entropies.append(s.spectral_entropy(f[e], Pxx[e]))
	entropies = np.array(entropies).reshape(-1,1)
	return Dataset(name=name, features=["ENTROPY"], data=entropies)
