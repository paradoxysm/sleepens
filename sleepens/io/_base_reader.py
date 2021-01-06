from abc import ABC, abstractmethod

def check_reader(reader):
	if not issubclass(type(reader), AbstractReader):
		raise ValueError("Object is not a Reader")

class AbstractReader(ABC):
	def __init__(self, verbose=0):
		self.standard = "Files"
		self.filetypes = [("Files", "*.*")]
		self.verbose = verbose

	@abstractmethod
	def read_data(self, filepath, *args, **kwargs):
		raise NotImplementedError("No read function implemented")

	@abstractmethod
	def read_labels(self, filepath, *args, **kwargs):
		raise NotImplementedError("No read function implemented")
