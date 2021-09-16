"""Percentile Mean Processor"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

from sleepens.io import DataObject, Dataset
from sleepens.utils.data import signal as s
from sleepens.utils import aggregate, determine_threshold, calculate_epochs, get_epoch

def epoched_variance(dataobject, epoch_size, name=None, sub_epoch_size=10,
				threshold='median', merge='sum'):
	"""
	Process the data in `dataobject`,
	divide the data into epochs of `epoch_size`, and
	compute mean of data above the `k`-th percentile.

	Parameters
	----------
	dataobject : DataObject
		DataObject for processing.

	epoch_size : float
		Number of seconds in each epoch.

	sub_epoch_size : int
		Number of sub-epochs in an epoch.

	threshold : {'mean', 'median', '%<k>', 'float'}
		Threshold to count as 'activity'.
		Must be one of:
		 - 'mean' : Mean of sub-epochs.
		 - 'median' : Median of sub-epochs.
		 - '%<k>' : <k>-th percentile of sub-epochs.
		 			<k> is a non-negative integer.
		 - float : Constant threshold.

	merge : {'mean', 'median', 'max', 'min', 'sum', 'count'}
	Determines merge method.
	Must be one of:
	 - 'mean' : Mean of variances above threshold.
	 - 'median' : Median of variances above threshold.
	 - 'max' : Max of variances above threshold.
	 - 'min' : Min of variances above threshold.
	 - 'sum' : Sum of variances above threshold.
	 - 'count' : Count of variances above threshold.

	Returns
	-------
	ds = Dataset
		Dataset with epoched variance at each epoch.
		Feature is named "EPOCHED VARIANCE"
	"""
	activities = []
	epoch_len, n_epochs = calculate_epochs(dataobject.data, dataobject.resolution, epoch_size)
	for i in range(n_epochs):
		epoch = get_epoch(dataobject.data, i, epoch_len)
		subepoch_var = []
		subepoch_len = epoch_len // sub_epoch_size
		for j in range(sub_epoch_size):
			subepoch = epoch[j*subepoch_len:(j+1)*subepoch_len]
			subepoch_var.append(np.var(subepoch))
		activities.append(subepoch_var)
	activities = np.array(activities)
	shape = activities.shape
	threshold = determine_threshold(activities.flatten(), threshold)
	activities = list(activities.reshape(shape))
	for i in range(n_epochs):
		above = np.where(activities[i] >= threshold)
		activities[i] = aggregate(activities[i][above], merge)
	activities = np.array(activities).reshape(-1,1)
	ds = Dataset(name=name, features=["EPOCHED VARIANCE"], data=activities)
	return ds
