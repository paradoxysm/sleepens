"""smrMAT I/O Interface"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from scipy.io import loadmat

from sleepens.io import DataObject, Dataset

name = "smrMAT"
standard = ".mat files exported by CED Spike2"
filetypes = [("MAT-files", "*.mat")]
type = "RAW"
tags = {'r'}

def read_data(filepath, channel):
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
	matfile = _load(filepath)
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

def read_labels(filepath, channel, map={}):
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
	matfile = _load(filepath)
	fields = [f for f in matfile.keys() if '_Ch' in f]
	channels = [matfile[field][0][0][0][0] for field in fields]
	if channel in channels:
		field = fields[channels.index(channel)]
		try:
			labels = matfile[field][0][0][7].flatten()[:-1]
			resolution = matfile[field][0][0][2][0][0]
			for k, v in map.items():
				labels[labels == k] = v
			labels = labels.astype(int)
		except Exception:
			raise FileNotFoundError("An error occurred extracting from channel")
	else:
		raise FileNotFoundError("Channel named " + channel + " not found. Instead found: " + str(channels))
	return DataObject(name=channel, data=labels, resolution=resolution)

def write(filepath, dataobjects):
	"""
	Write the dataset to a file.

	Parameters
	----------
	filepath : path
		Path to the .mat file to write.

	dataobjects : array-like of DataObject, shape=(n_channels,)
		DataObjects to write to the file. DataObjects
		with resolution set to -1 are assumed as labels.
	"""
	raise NotImplementedError("smrMAT cannot write to files")

def _load(filepath):
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
