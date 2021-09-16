"""Percentile Mean Processor"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.io import DataObject, Dataset
from sleepens.utils.data import signal as s
from sleepens.utils import calculate_epochs, get_epoch

def rms(dataobject, epoch_size, name=None):
	"""
	Process the data in `dataobject`,
	divide the data into epochs of `epoch_size`, and
	compute the root-mean-square at each epoch.

	Parameters
	----------
	dataobject : DataObject
		DataObject for processing.

	epoch_size : int
		Number of seconds in each epoch.

	Returns
	-------
	ds = Dataset
		Dataset with root-mean-square at each epoch.
		Feature is named "RMS"
	"""
	rms = []
	epoch_len, n_epochs = calculate_epochs(dataobject.data, dataobject.resolution, epoch_size)
	for i in range(n_epochs):
		epoch = get_epoch(dataobject.data, i, epoch_len)
		rms.append(s.rms(epoch))
	rms = np.array(rms).reshape(-1,1)
	return Dataset(name=name, features=["RMS"], data=rms)
