"""Dataset"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import xlrd
import xlwt
import os
import numpy as np

from sleepens.utils import is_float, is_int

class Dataset:
	"""
	Dataset used between for Sleep Ensemble.

	Parameters
	----------
	name : string, default="noname"
		Name of the dataset. When the dataset is exported,
		this is the name of the file.

	features : array-like, shape=(n_features), default=[]
		The names of the features of the data.
		Length of `features` must match the number of columns
		in `data`.

	label_names : array_like, shape=(n_labels), default=[]
		The names of the labels of the data.
		Length of `label_names` must match the number of columns
		in `labels`.

	data : array_like, shape=(n_samples, n_features), default=[]
		The data for this Dataset.

	labels : array_like, shape=(n_samples, n_labels) default=[]
		The labels for this Dataset.
	"""
	def __init__(self, name="noname", features=[], label_names=[],
					data=[], labels=[]):
		self.name = name
		self.features = np.array(features)
		self.label_names = np.array(label_names)
		self.data = np.array(data)
		self.labels = np.array(labels)

	def read(self, files, sheets=None, cols=[], label_cols=[]):
		"""
		Read from the list of .xls files given into a single Dataset.
		.xls files should be formatted such that all required data
		is found in a single sheet, with a header row and contiguous data.

		Parameters
		----------
		files : list of string
			The list of filepaths to .xls data files to read.

		sheets : list, None, default=None
			The list of sheet names to use. If None, imports
			data in the first sheet only. If int, opens the
			i-th sheet. If string, opens the sheet by the given name.
			For int or string usage, `sheets` must be in a list where
			the i-th element corresponds to the i-th file.

		cols : list, default=[]
			Data columns to import. Zero-indexed.

		label_cols : list, default=[]
			Label columns to import. Zero-indexed.
		"""
		features, data, labels = [], [], []
		for i in range(len(files)):
			ds = xlrd.open_workbook(files[i])
			if sheets is None : data_sheet = ds.sheet_by_name(ds.sheet_names()[0])
			elif isinstance(sheets[i], int) : data_sheet = ds.sheet_by_name(ds.sheet_names()[sheets[i]])
			elif isinstance(sheets[i], str) : data_sheet = ds.sheet_by_name(sheets[i])
			header = np.array(data_sheet.row_values(0))
			self.label_names = header[label_cols]
			self.features = header[cols]
			rows = data_sheet.nrows
			for row in range(1, rows):
				data_row = []
				data_label = []
				for col in cols:
					data_row.append(read_cell(data_sheet, row, col))
				for col in label_cols:
					data_label.append(read_cell(data_sheet, row, col))
				data.append(data_row)
				labels.append(data_label)
		self.data = np.array(data)
		self.labels = np.array(labels)

	def write(self, loc):
		"""
		Write the dataset into the given folder location.
		The dataset will export as a .xls file with the
		filename as the name of the dataset.

		Parameters
		----------
		loc : string
			Folder location to export the dataset.
		"""
		header = np.concatenate((self.features, self.label_names))
		header = header.reshape(1, len(header))
		if len(self.labels) > 0:
			labels = self.labels.reshape(-1,len(self.label_names))
			contents = np.concatenate((self.data, labels), axis=1)
		else:
			contents = self.data
		if header.shape[1] != contents.shape[1]:
			raise ValueError("Dimensions of header and data do not match")
		filepath = loc + '/' + self.name + '.xls'
		if os.path.isfile(filepath) : os.remove(filepath)
		wb = xlwt.Workbook()
		ws = wb.add_sheet("Results")
		output = np.concatenate((header, contents))
		for row in range(output.shape[0]):
			for col in range(output.shape[1]):
				ws.write(row, col, output[row][col])
		wb.save(filepath)

	def split(self, name="noname", rows=None, cols=None, label=None):
		"""
		Create a new Dataset with a subset of the current Dataset.

		Parameters
		----------
		name : string, default="noname"
			The name of the new Dataset.

		rows : int, list of indices, None, default=None
			Rows of data to use. If None, take all rows.

		cols : int, list of indices, None, default=None
			Columns of data to use. If None, take all columns.

		label : int, list of indices, None, default=None
			Labels to use. If None, take all labels.

		Returns
		-------
		ds : Dataset
			The newly created Dataset.
		"""
		data = self.data if rows is None else self.data[rows]
		data = self.data if cols is None else data[:,cols]
		features = self.features if cols is None else self.features[cols]
		labels = self.labels if rows is None else self.labels[rows]
		labels = self.labels if label is None else labels[:,label]
		label_names = self.label_names if label is None else self.label_names[label]
		return Dataset(name, features, label_names, data, labels)

	def append(self, ds):
		"""
		Append a Dataset to this Dataset.
		The features and label names of the appending Dataset
		must match this Dataset. The number of columns for
		data and labels must also match.

		Parameters
		----------
		ds : Dataset
			The Dataset to append.

		Returns
		-------
		self : Dataset
			Returns this Dataset.
		"""
		features = ds.features
		if np.any(features != self.features):
			raise ValueError("Features do not match. Looking for " + \
								self.features + ". Got " + features + ".")
		data = ds.data
		if data.shape[1] != self.data.shape[1]:
			raise ValueError("Data columns do not match. Looking for " + \
								self.data.shape[1] + " columns. Got " + \
								data.shape[1] + ".")
		label_names = ds.label_names
		if np.any(label_names != self.label_names):
			raise ValueError("Label names do not match. Looking for " + \
								self.label_names + ". Got " + label_names + ".")
		labels = ds.labels
		if labels.shape[1] != self.labels.shape[1]:
			raise ValueError("Label columns do not match. Looking for " + \
								self.labels.shape[1] + " columns. Got " + \
								labels.shape[1] + ".")
		self.data = np.concatenate((self.data, data))
		self.labels = np.concatenate((self.labels, labels))
		return self

	def concatenate(self, ds, overlap=True):
		"""
		Concatenate a Dataset to this Dataset.
		The number of samples in both datasets must match.

		Parameters
		----------
		ds : Dataset
			The Dataset to append.

		overlap : bool, default=True
			Determine how to handle duplicate features or labels.
			When True, only use the features/labels copy in this
			Dataset. When False, use both.

		Returns
		-------
		self : Dataset
			Returns this Dataset.
		"""
		if len(self.data) == 0:
			self.features = ds.features
			self.data = ds.data
		elif len(self.data) == len(ds.data):
			if overlap : feature_overlap = np.isin(ds.features, self.features, invert=True)
			else : feature_overlap = np.arange(ds.features.shape[0])
			self.features = np.concatenate((self.features, ds.features[feature_overlap]))
			self.data = np.concatenate((self.data, ds.data[:,feature_overlap]), axis=1)
		if len(self.labels) == 0:
			self.label_names = ds.label_names
			self.labels = ds.labels
		elif len(self.labels) == len(ds.labels):
			if overlap : label_overlap = np.isin(ds.label_names, self.label_names, invert=True)
			else : label_overlap = np.arange(ds.label_names.shape[0])
			self.label_names = np.concatenate((self.label_names, ds.label_names[label_overlap]))
			self.labels = np.concatenate((self.labels, ds.labels[:,label_overlap]), axis=1)
		else : raise ValueError("Label lengths do not match")
		return self

	def clean(self):
		"""
		Clean the dataset for mismatched data and label lengths.
		Truncates the longer of data or labels to match.

		Returns
		-------
		self : Dataset.
			Returns this Dataset.
		"""
		if len(self.label_names) > 0:
			length = min([len(self.data), len(self.labels)])
			self.data = self.data[:length]
			self.labels = self.labels[:length]
		else:
			self.labels = np.array([])
		return self

"""
MISCELLANEOUS FUNCTIONS
"""
def read_cell(data_sheet, row, col):
	"""
	Read a cell in a .xls file and convert to
	int or float where possible.

	Parameters
	----------
	data_sheet : Sheet
		The Sheet to read the data.

	row : int
		The row to read at.

	col : int
		The column to read at.

	Returns
	-------
	cell : string, int, float
		The data found at the given cell.
	"""
	try:
		cell = data_sheet.cell_value(row, col)
	except:
		print(row, col)
	if is_float(cell) : cell = float(cell)
	elif is_int(cell) : cell = int(cell)
	return cell
