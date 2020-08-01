"""Spectral Band Processor"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.io import Dataset
from sleepens.utils.data import signal as s

def spectral_band(ds, name=None, bands=None, merge='sum'):
	"""
	Process the epoched power spectral density
	and compute power at a specific frequency
	band.

	Parameters
	----------
	f : ndarray
		List of arrays of frequencies for each epoch.

	Pxx : ndarray
		List of power spectral densities of `data` at
		each epoch.

	params : dict
		Dictionary of all band parameters:
		 - BANDS :  Dictionary of tuple intervals for bands.
		 - BAND_MERGE : Method to merge power spectral densities
		 				within bands. Select from
						{'mean', 'max', 'min', 'sum'}.

	Returns
	-------

	"""
	f, Pxx = ds.features, ds.data
	if bands is None : bands = {'ALL': (np.min(f), np.max(f))}
	names = list(bands.keys())
	data = []
	for e in range(min([len(f), len(Pxx)])):
		band = s.compute_bands(f[e], Pxx[e], bands, merge)
		band = [band[k] for k in names]
		data.append(band)
	data = np.array(data)
	return Dataset(name=name, features=names, data=data)
