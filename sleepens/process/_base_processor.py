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
	def __init__(self, verbose=0):
		self.verbose = verbose

	@abstractmethod
	def process(self, data, labels, name):
		raise NotImplementedError("No process function implemented")
