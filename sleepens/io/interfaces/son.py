"""SON I/O Interface"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from copy import deepcopy
from pathlib import Path
from sonpy import lib as sp

from sleepens.io import DataObject, Dataset

name = "SON"
standard = ".smr files exported by CED Spike2"
filetypes = [("SMR File", "*.smr")]
type = "RAW"
tags = {'r', 'w'}

def read_data(filepath, channel):
	"""
	Read the data file at a specific data channel.

	Parameters
	----------
	filepath : path
		Path to the .smr file.

	channel : str
		Name of the channel in the .smr file.

	Returns
	-------
	dataobject : DataObject
		The DataObject containing the data from
		the specific channel.
	"""
	sonfile = _load(filepath)
	channel_idx = _get_channel(sonfile, channel)
	max_time = sonfile.ChannelMaxTime(channel_idx)
	divide = sonfile.ChannelDivide(channel_idx)
	n = int(max_time / divide)
	data = sonfile.ReadFloats(channel_idx, n, 0, max_time)
	resolution = sonfile.GetTimeBase() * divide
	del sonfile
	return DataObject(name=channel, data=data, resolution=resolution, divide=divide)

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
		set of integers. Not used with SON
		as codes are already in integers.

	Returns
	-------
	dataobject : DataObject
		The DataObject containing the labels from
		the specific channel.
	"""
	sonfile = _load(filepath)
	channel_idx = _get_channel(sonfile, channel)
	max_time = sonfile.ChannelMaxTime(channel_idx)
	divide = sonfile.ChannelDivide(channel_idx)
	resolution = sonfile.GetTimeBase() * divide
	labels = sonfile.ReadTextMarks(channel_idx, 2, 0, max_time)
	if len(labels) > 1:
		interval = labels[1].Tick - labels[0].Tick
		n = int(max_time / interval)
		labels = sonfile.ReadTextMarks(channel_idx, n, 0, max_time)
	for i in range(len(labels)):
		labels[i] = labels[i].Code1
	labels = np.array(labels).flatten()
	del sonfile
	return DataObject(name=channel, data=labels, resolution=resolution, divide=divide)

def write(filepath, dataobjects, labels, map={}, epoch_size=5):
	"""
	Write the dataset to a file.

	Parameters
	----------
	filepath : path
		Path to the .smr file to write.

	dataobjects : array-like of DataObject, shape=(n_channels,)
		DataObjects to write to the file.

	labels : DataObject
		Labels to write to the file.

	map : dict, default={}
		Mapping label integers to some set of label values.

	epoch_size : int, default=5
		Number of seconds in an epoch for labels.
	"""
	sonfile = sp.SonFile(filepath)
	if sonfile.GetOpenError() != 0:
		raise RunTimeError("Error writing .smr file:", sp.GetErrorString(sonfile.GetOpenError()))
	if not sonfile.CanWrite():
		raise RunTimeError("File is read only")
	num_channels = len(dataobjects)
	sonfile.SetTimeBase(labels.resolution)
	for i in range(num_channels):
		sonfile = _write_data(sonfile, dataobjects[i], i)
	tickbase = int(epoch_size / sonfile.GetTimeBase())
	if labels is not None:
		sonfile = _write_labels(sonfile, labels, num_channels, map, tickbase)
	del sonfile

def _write_data(sonfile, dataobject, channel_idx):
	"""
	Write an Adc channel into the SonFile.

	Parameters
	----------
	sonfile : SonFile
		SonFile object.

	dataobject : DataObject
		DataObject for writing.

	channel_idx : int
		The index of the channel to write to.

	Returns
	-------
	sonfile : SonFile
		SonFile file object.
	"""
	sonfile.SetWaveChannel(channel_idx, dataobject.divide, sp.DataType.RealWave)
	sonfile.SetChannelTitle(channel_idx, dataobject.name)
	sonfile.SetChannelUnits(channel_idx, 'volts')
	sonfile.SetChannelScale(channel_idx, 1.)
	sonfile.SetChannelOffset(channel_idx, 0.)
	sonfile.SetChannelYRange(channel_idx, -5., 5.)
	sonfile.WriteFloats(channel_idx, dataobject.data, 0)
	return sonfile

def _write_labels(sonfile, dataobject, channel_idx, map, tickbase):
	"""
	Write a TextMark channel into the SonFile.

	Parameters
	----------
	sonfile : SonFile
		SonFile object.

	dataobject : DataObject
		DataObject for writing.

	channel_idx : int
		The index of the channel to write to.

	map : dict, default={}
		Mapping label integers to some set of label values.

	epoch_size : int, default=5
		Number of seconds in an epoch for labels.

	Returns
	-------
	sonfile : SonFile
		SonFile file object.
	"""
	labels = deepcopy(dataobject.data)
	codes = deepcopy(dataobject.data)
	labels = [map[i] for i in labels]
	textmarks = np.empty(len(labels), dtype=sp.TextMarker)
	max_text_size = max(len(s) for s in labels) + 1
	sonfile.SetTextMarkChannel(channel_idx, 2, max_text_size)
	sonfile.SetChannelTitle(channel_idx, dataobject.name)
	tick = 0
	for i in range(len(labels)):
		textmarks[i] = sp.TextMarker(labels[i], nTick=tick, nCode1=codes[i])
		tick += tickbase
	sonfile.WriteTextMarks(channel_idx, textmarks)
	return sonfile

def _load(filepath):
	"""
	Attempt to load the .smr file.

	Parameters
	----------
	filepath : path
		Path to the .smr file.

	Returns
	-------
	sonfile : SonFile
		SonFile file object.
	"""
	sonfile = sp.SonFile(filepath, True)
	if sonfile.GetOpenError() != 0:
		raise FileNotFoundError("Error opening .smr file:", sp.GetErrorString(sonfile.GetOpenError()))
	return sonfile

def _get_channel(sonfile, channel):
	"""
	Find the specified channel index within the SonFile.

	Parameters
	----------
	sonfile : SonFile
		SonFile file object.

	channel : str
		Name of the channel in the .smr file.

	Returns
	-------
	channel_idx : int
		The index of the specified channel.
	"""
	max_channels = sonfile.MaxChannels()
	channels = []
	for i in range(max_channels):
		if sonfile.GetChannelTitle(i) == channel:
			return i
		elif sonfile.ChannelType(i) != sp.DataType.Off:
			channels.append(sonfile.GetChannelTitle(i))
	raise FileNotFoundError("Channel named " + channel + " not found. Instead found: " + str(channels))
