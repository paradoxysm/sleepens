"""smrMAT Reader"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from scipy.io import loadmat

from sleepens.io import DataObject

standard = ".mat files exported by Spike2 v7"
filetypes = [("MAT-files", "*.mat")]

def read_smrMAT(filepath, c, name=None, score=False, map={}):
	"""
	Read the data file at a specific channel.

	Parameters
	----------
	filepath : path
		Path to the .mat file.

	c : str
		Name of the channel in the .mat file.

	name : str, default=None
		Desired name of the resulting DataObject.
		If None, use `c`.

	score : bool, default=False
		Determines if the channel is a score
		channel, which reads differently from
		data channels.

	map : dict, default={}
		Mapping the score values to some
		set of integers.

	Returns
	-------
	dataobject : DataObject
		The DataObject containing the data from
		the specific channel.
	"""
	matfile = _load(filepath)
	fields = [f for f in matfile.keys() if '_Ch' in f]
	channels = [matfile[field][0][0][0][0] for field in fields]
	if score and c in channels:
		field = fields[channels.index(c)]
		data, resolution = _score_read(matfile[field][0][0], map)
	elif c in channels:
		field = fields[channels.index(c)]
		data, resolution = _data_read(matfile[field][0][0])
	else:
		raise FileNotFoundError("Channel named " + c + " not found. Instead found: " + str(channels))
	if name is None:
		name = c
	return DataObject(name=name, data=data, resolution=resolution)

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

def _score_read(channel, map):
	"""
	Read the score data from the channel,
	translated according to the given map.

	If the map is an empty dict, no translation
	is done.

	Parameters
	----------
	channel : list
		Channel containing score data.

	map : dict
		Mapping the score values to some
		set of integers.

	Returns
	-------
	data : list
		The score data.

	resolution : float
		The resolution of the data.
	"""
	try:
		data = channel[7].flatten()[:-1]
		for k, v in map.items():
			data[data == k] = v
		data = data.astype(int)
		resolution = channel[2][0][0]
		return data, resolution
	except Exception:
		raise FileNotFoundError("An error occurred extracting from channel")

def _data_read(channel):
	"""
	Read the signal data from the channel.

	Parameters
	----------
	channel : list
		Channel containing score data.

	Returns
	-------
	data : list
		The score data.

	resolution : float
		The resolution of the data.
	"""
	try:
		data = channel[8].flatten()
		resolution = channel[2][0][0]
		return data, resolution
	except Exception:
		raise FileNotFoundError("An error occurred extracting from channel")
