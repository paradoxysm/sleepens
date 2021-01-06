from abc import ABC, abstractmethod

class AbstractAddon(ABC):
	def __init__(self, verbose=0):
		self.verbose = verbose

	@abstractmethod
	def addon(self, Y_hat, p, *args, **kwargs):
		raise NotImplementedError("No addon function implemented")
