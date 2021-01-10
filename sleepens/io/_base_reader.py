"""Base Reader"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

from abc import ABC, abstractmethod

def check_reader(reader):
	"""
	Checks that `reader` is a subclass of
	AbstractReader. Raises an error if
	it is not.

	Parameters
	----------
	reader : object
		Object to check.
	"""
	if not issubclass(type(reader), AbstractReader):
		raise ValueError("Object is not a Reader")

class AbstractReader(ABC):
	"""
	Base Reader Superclass. Files are given to the Reader
	and read into a dataobject.

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
	name = "Abstract"
	standard = "Files"
	filetypes = [("Files", "*.*")]

	def __init__(self, verbose=0):
		self.verbose = verbose

	@abstractmethod
	def read_data(self, filepath, *args, **kwargs):
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
		raise NotImplementedError("No read function implemented")

	@abstractmethod
	def read_labels(self, filepath, *args, **kwargs):
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
		raise NotImplementedError("No read function implemented")
