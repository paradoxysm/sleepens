"""Epoched Processor"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from math import floor

from sleepens.io import DataObject

def epochify(data, resolution, epoch_size):
	"""
	Calculate the epoch length and number of epochs
	in the data based on `epoch_size`.

	Parameters
	----------
	dataobject : DataObject
		DataObject for processing.

	epoch_size : int
		Number of seconds in each epoch.

	Returns
	-------
	epoch_len : int
		Number of samples in each epoch.

	n_epochs : int
		Number of full epochs in the data.
	"""
	epoch_len = floor(epoch_size / resolution)
	n_epochs = floor(len(data) / epoch_len)
	return epoch_len, n_epochs

def get_epoch(data, i, epoch_len):
	"""
	Get the data corresponding to the `i`-th
	epoch.

	Parameters
	----------
	dataobject : DataObject
		DataObject for processing.

	i : int
		The epoch to collect.

	epoch_len : int
		Number of samples in each epoch.

	Returns
	-------
	data : ndarray, shape=(epoch_len,)
		The data of the `i`-th epoch.
	"""
	start = i * epoch_len
	end = (i + 1) * epoch_len
	return data[start:end]
