"""smrMAT Reader"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from scipy.io import loadmat

from sleepens.io import DataObject
from sleepens.io._base_reader import AbstractReader

class smrMATReader(AbstractReader):
	"""
	Reader for .mat files exported by Spike2 v7.

	Parameters
	----------
	verbose : int, default=0
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.

	Attributes
	----------
	name : string
		Name of the Reader.

	standard : string
		The name of accepted filetypes the Reader accepts.

	filetypes : list of tuple
		The accepted filetypes of the Reader.
		Each tuple pair consists of the filetype name
		followed by regex for the filetype.
	"""
	name = "smrMAT"
	standard = ".mat files exported by Spike2 v7"
	filetypes = [("MAT-files", "*.mat")]

	def __init__(self, verbose=0):
		self.verbose = verbose

	def read_data(self, filepath, channel):
		"""
		Read the data file at a specific data channel.

		Parameters
		----------
		filepath : path
			Path to the .mat file.

		channel : str
			Name of the channel in the .mat file.

		Returns
		-------
		dataobject : DataObject
			The DataObject containing the data from
			the specific channel.
		"""
		matfile = self._load(filepath)
		fields = [f for f in matfile.keys() if '_Ch' in f]
		channels = [matfile[field][0][0][0][0] for field in fields]
		if channel in channels:
			field = fields[channels.index(channel)]
			try:
				data = matfile[field][0][0][8].flatten()
				resolution = matfile[field][0][0][2][0][0]
			except Exception:
				raise FileNotFoundError("An error occurred extracting from channel")
		else:
			raise FileNotFoundError("Channel named " + channel + " not found. Instead found: " + str(channels))
		return DataObject(name=channel, data=data, resolution=resolution)

	def read_labels(self, filepath, channel, map={}):
		"""
		Read the data file at a specific label channel.

		Parameters
		----------
		filepath : path
			Path to the .mat file.

		channel : str
			Name of the channel in the .mat file.

		map : dict, default={}
			Mapping the label values to some
			set of integers.

		Returns
		-------
		dataobject : DataObject
			The DataObject containing the labels from
			the specific channel.
		"""
		matfile = self._load(filepath)
		fields = [f for f in matfile.keys() if '_Ch' in f]
		channels = [matfile[field][0][0][0][0] for field in fields]
		if channel in channels:
			field = fields[channels.index(channel)]
			try:
				labels = matfile[field][0][0][7].flatten()[:-1]
				for k, v in map.items():
					labels[labels == k] = v
				labels = labels.astype(int)
				resolution = matfile[field][0][0][2][0][0]
			except Exception:
				raise FileNotFoundError("An error occurred extracting from channel")
		else:
			raise FileNotFoundError("Channel named " + channel + " not found. Instead found: " + str(channels))
		return DataObject(name=channel, data=labels, resolution=resolution)

	def _load(self, filepath):
		"""
		Attempt to load the .mat file.

		Parameters
		----------
		filepath : path
			Path to the .mat file.

		Returns
		-------
		matfile : dict
			Dictionary with variable names as keys
			and matrices as values.
		"""
		try:
			matfile = loadmat(filepath)
		except:
			raise FileNotFoundError("No such file or directory: " + filepath)
		return matfile
