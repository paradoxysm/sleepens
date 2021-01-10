"""Base Processor"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

from abc import ABC, abstractmethod

def check_processor(processor):
	"""
	Checks that `processor` is a subclass of
	AbstractProcessor. Raises an error if
	it is not.

	Parameters
	----------
	processor : object
		Object to check.
	"""
	if not issubclass(type(processor), AbstractProcessor):
		raise ValueError("Object is not a Processor")

class AbstractProcessor(ABC):
	"""
	Base Processor Superclass. Data is fed to a
	processor for preprocessing and feature extraction.

	Parameters
	----------
	verbose : int, default=0
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.
	"""
	def __init__(self, verbose=0):
		self.verbose = verbose

	@abstractmethod
	def process(self, data, labels, name):
		"""
		Process the data and produce a Dataset
		with the given name and labels, if provided.

		Parameters
		----------
		data : array-like
			The data to process. The specific data structure
			can vary depending on the Processor implementation.

		labels : array-like, shape=(n_samples,)
			The labels associated with the data.
			If None, does not add labels.

		name : string
			Name of the resulting dataset. If None,
			uses the default Dataset name, "noname"

		Returns
		-------
		ds : Dataset
			The processed dataset.
		"""
		raise NotImplementedError("No process function implemented")
