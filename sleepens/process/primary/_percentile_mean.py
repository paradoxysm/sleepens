"""Percentile Mean Processor"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.io import DataObject, Dataset
from sleepens.utils.data import signal as s
from sleepens.utils import calculate_epochs, get_epoch

def percentile_mean(dataobject, epoch_size, name=None, k=50):
	"""
	Process the data in `dataobject`,
	divide the data into epochs of `epoch_size`, and
	compute mean of data above the `k`-th percentile.

	Parameters
	----------
	dataobject : DataObject
		DataObject for processing.

	epoch_size : int
		Number of seconds in each epoch.

	k : float
		Percentile threshold.

	Returns
	-------
	ds = Dataset
		Dataset with percentile mean at each epoch.
		Feature is named "<k> PERCENTILE MEAN"
	"""
	percentile_mean = []
	epoch_len, n_epochs = calculate_epochs(dataobject.data, dataobject.resolution, epoch_size)
	for i in range(n_epochs):
		epoch = get_epoch(dataobject.data, i, epoch_len)
		percentile_mean.append(s.percentile_mean(epoch, k))
	percentile_mean = np.array(percentile_mean).reshape(-1,1)
	return Dataset(name=name, features=[str(k) + " PERCENTILE MEAN"], data=percentile_mean)
