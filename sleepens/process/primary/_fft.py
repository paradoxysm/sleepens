"""Fast Fourier Transform Processor"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.io import DataObject, Dataset
from sleepens.utils.data import signal as s
from sleepens.utils import calculate_epochs, get_epoch

def fft(dataobject, epoch_size, name=None, nperseg_factor=1,
				noverlap_factor=0.1, detrend='constant'):
	"""
	Process the data in `dataobject`,
	divide the data into epochs of `epoch_size`, and
	compute the Fast Fourier Transform (FFT).

	Parameters
	----------
	dataobject : DataObject
		DataObject for processing.

	epoch_size : int
		Number of seconds in each epoch.

	nperseg_factor : float (0, inf)
		Multiplied by data resolution to
		calculate the NPERSEG value.

	noverlap_factor : float (0, inf)
		Multiplied by NPERSEG to calculate
		the NOVERLAP value.

	detrend : str or func or False, default='constant'
		Method to detrend each segment.

	Returns
	-------
	ds = Dataset
		Dataset with frequencies as features
		and power spectral densities at each epoch.
	"""
	f, Pxx = [], []
	epoch_len, n_epochs = calculate_epochs(dataobject.data, dataobject.resolution, epoch_size)
	for i in range(n_epochs):
		epoch = get_epoch(dataobject.data, i, epoch_len)
		fs = 1/dataobject.resolution
		nperseg = fs * nperseg_factor
		noverlap = nperseg * noverlap_factor
		fi, Pxxi = s.welch(epoch, fs=fs, nperseg=nperseg, noverlap=noverlap,
								detrend=detrend)
		f.append(fi)
		Pxx.append(Pxxi)
	return Dataset(name=name, features=f, data=Pxx)
