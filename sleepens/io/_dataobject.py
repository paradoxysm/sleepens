"""Data Object"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np

class DataObject:
	"""
	DataObject is a container to store some signal data.

	Parameters
	----------
	name : string, default=""
		Name of the DataObject.

	data : ndarray, default=()
		Signal data contained in the DataObject.

	resolution : float, default=-1
		Resolution of the signal data.

	length : int, default=-1
		Number of samples in the signal data.

	divide : int, default=-1
		Number of clock ticks per sample.

	Attributes
	----------
	indices : ndarray
		Array of indices corresponding to the signal data.
	"""
	def __init__(self, name="", data=np.zeros(0), resolution=-1, divide=1):
		self.name = name
		self.data = data
		self.indices = np.arange(len(data))
		self.resolution = resolution
		self.length = len(self.data)
		self.divide = divide

	def __str__(self):
		return type(self).__name__ + ": " + self.name

	def _get(self, arr, i, k=None):
		"""
		Return a subarray of `arr` at index `i`
		if `k` is None or from index `i`
		up to (non-exclusive) `k`.

		Parameters
		----------
		arr : array-like
			Array to be sliced.

		i : int
			Index of the first element of the returned slice.

		k : int or None, default=None
			End of the sliced subarray.

		Returns
		-------
		subarray : array-like
			Subarray of `arr`.
		"""
		if len(arr) == 0:
			return False
		if isinstance(k, int):
			return arr[i:k]
		else:
			return arr[i]

	def get_data(self, i, k=None):
		"""
		Get a subarray of the signal data from index
		`i` up to (non-exclusive) `k` or the end of the
		list.

		Parameters
		----------
		i : int
			Index of the first element of the returned slice.

		k : int or None, default=None
			End of the sliced subarray. End of the list if
			None.

		Returns
		-------
		subarray : array-like
			Subarray of data.
		"""
		return self._get(self.data, i, k)

	def get_indices(self, i, k=None):
		"""
		Get a subarray of indices from index
		`i` up to (non-exclusive) `k` or the end of the
		list.

		Parameters
		----------
		i : int
			Index of the first element of the returned slice.

		k : int or None, default=None
			End of the sliced subarray. End of the list if
			None.

		Returns
		-------
		subarray : array-like
			Subarray of indices.
		"""
		return self._get(self.indices, i, k)

	def process(self, function, arg=None):
		"""
		Process the signal data by a given function
		returning a new DataObject with processed data.

		Parameters
		----------
		function : function or None
			Defined function to process signal data.
			If None, return itself.

		arg : list or None, default=None
			Additional arguments for the defined function.

		Returns
		-------
		dataobject : DataObject
			DataObject with the signal data after process
			modification.
		"""
		if function is None:
			return self
		elif arg is None:
			data = function(self.data)
		else:
			data = function(self.data, arg)
		return DataObject(name=self.name, data=data, resolution=self.resolution)
