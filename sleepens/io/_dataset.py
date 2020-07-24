import xlrd
import xlwt
import os
import numpy as np

from sleepens.utils.misc import is_float, is_int

class Dataset:
	def __init__(self, name=None, features=None, label_names=None,
					data=None, labels=None):
		self.name = "noname" if name is None else str(name)
		self.features = np.array([]) if features is None else np.array(features)
		self.label_names = np.array([]) if label_names is None else np.array(label_names)
		self.data = np.array([]) if data is None else np.array(data)
		self.labels = np.array([]) if labels is None else np.array(labels)

	def read(self, files, sheets=None, cols=[], label_cols=[]):
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
		header = np.concatenate((self.features, self.label_names))
		header = header.reshape(1, len(header))
		contents = np.concatenate((self.data, self.labels), axis=1)
		if header.shape[1] != contents.shape[1]:
			raise ValueError("Dimensions of header and data do not match")
		filepath = loc + '/' + self.name + '.xls'
		if os.path.isfile(filepath) : os.remove(filepath)
		wb = xlwt.Workbook()
		ws = wb.add_sheet(self.name)
		output = np.concatenate((header, contents))
		for row in range(output.shape[0]):
			for col in range(output.shape[1]):
				ws.write(row, col, output[row][col])
		wb.save(filepath)

	def split(self, name, rows=None, cols=None, label=None):
		data = self.data if rows is None else self.data[rows]
		data = self.data if cols is None else data[:,cols]
		features = self.features if cols is None else self.features[cols]
		labels = self.labels if rows is None else self.labels[rows]
		labels = self.labels if label is None else labels[:,label]
		label_names = self.label_names if label is None else self.label_names[label]
		return Dataset(name, features, label_names, data, labels)

	def append(self, ds):
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
		length = min([len(self.data), len(self.labels)])
		self.data = self.data[:length]
		self.labels = self.labels[:length]
		return self

"""
MISCELLANEOUS FUNCTIONS
"""
def read_cell(data_sheet, row, col):
	cell = data_sheet.cell_value(row, col)
	if is_float(cell) : cell = float(cell)
	elif is_int(cell) : cell = int(cell)
	return cell
