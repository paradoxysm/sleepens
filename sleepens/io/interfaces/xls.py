"""Excel Worksheet I/O Interface"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
import xlrd, xlwt
import os
from pathlib import Path

from sleepens.io import Dataset
from sleepens.utils import is_float, is_int

name = "XLS"
standard = ".xls dataset files"
filetypes = [("Microsoft Excel Worksheet", "*.xls")]
type = "DATASET"
tags = {'r', 'w'}

def read_data(filepath, cols, sheet=None, ):
	"""
	Read the data file at the specified columns.

	Parameters
	----------
	filepath : path
		Path to the .xls file.

	cols : array-like of int, shape=(n_features,)
		Indices of columns to read.

	sheet : int, str, None, default=None
		The sheet to use. If None, imports
		data in the first sheet only. If int, opens the
		i-th sheet. If string, opens the sheet by the given name.

	Returns
	-------
	dataset : Dataset
		The Dataset containing the data from
		the specified columns.
	"""
	ds = xlrd.open_workbook(filepath)
	if sheet is None : data_sheet = ds.sheet_by_name(ds.sheet_names()[0])
	elif isinstance(sheet, int) : data_sheet = ds.sheet_by_name(ds.sheet_names()[sheet])
	elif isinstance(sheet, str) : data_sheet = ds.sheet_by_name(sheet)
	header = np.array(data_sheet.row_values(0))
	features = header[cols]
	data = []
	rows = data_sheet.nrows
	for row in range(1, rows):
		data_row = []
		for col in cols:
			data_row.append(_read_cell(data_sheet, row, col))
		data.append(data_row)
	name = Path(filepath).stem
	return Dataset(name=name, features=features, data=data)

def read_labels(filepath, cols, sheet=None, map={}):
	"""
	Read the data file at the specified columns.

	Parameters
	----------
	filepath : path
		Path to the .xls file.

	cols : array-like of int, shape=(n_features,)
		Indices of columns to read.

	sheet : int, str, None, default=None
		The sheet to use. If None, imports
		data in the first sheet only. If int, opens the
		i-th sheet. If string, opens the sheet by the given name.

	map : dict, default={}
		Mapping the label values to some
		set of integers.

	Returns
	-------
	dataset : Dataset
		The Dataset containing the labels from
		the specified columns.
	"""
	ds = xlrd.open_workbook(filepath)
	if sheet is None : data_sheet = ds.sheet_by_name(ds.sheet_names()[0])
	elif isinstance(sheet, int) : data_sheet = ds.sheet_by_name(ds.sheet_names()[sheet])
	elif isinstance(sheet, str) : data_sheet = ds.sheet_by_name(sheet)
	header = np.array(data_sheet.row_values(0))
	label_names = header[cols]
	labels = []
	rows = data_sheet.nrows
	for row in range(1, rows):
		label_row= []
		for col in cols:
			cell = _read_cell(data_sheet, row, col)
			if map : label_row.append(map[cell])
			else : label_row.append(cell)
		labels.append(label_row)
	name = Path(filepath).stem
	return Dataset(name=name, label_names=label_names, labels=labels)

def write(filepath, dataset):
	"""
	Write the dataset to a file.

	Parameters
	----------
	filepath : path
		Path to the .xls file to write.

	dataset : Dataset
		Dataset to write.
	"""
	header = np.concatenate((dataset.features, dataset.label_names))
	header = header.reshape(1, len(header))
	if len(dataset.labels) > 0:
		labels = dataset.labels.reshape(-1,len(dataset.label_names))
		contents = np.concatenate((dataset.data, labels), axis=1)
	else:
		contents = dataset.data
	if header.shape[1] != contents.shape[1]:
		raise ValueError("Dimensions of header and data do not match")
	if os.path.isfile(filepath) : os.remove(filepath)
	wb = xlwt.Workbook()
	ws = wb.add_sheet("Results")
	output = np.concatenate((header, contents))
	for row in range(output.shape[0]):
		for col in range(output.shape[1]):
			ws.write(row, col, output[row][col])
	wb.save(filepath)

def _read_cell(data_sheet, row, col):
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
