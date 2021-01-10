"""Base Addon"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

from abc import ABC, abstractmethod

def check_addon(addon):
	"""
	Checks that `addon` is a subclass of
	AbstractAddon. Raises an error if
	it is not.

	Parameters
	----------
	addon : object
		Object to check.
	"""
	if not issubclass(type(addon), AbstractAddon):
		raise ValueError("Object is not a Addon")

class AbstractAddon(ABC):
	"""
	Base Addon Superclass. Predictions are given
	for final post-processing.

	Parameters
	----------
	verbose : int, default=0
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.
	"""
	def __init__(self, verbose=0):
		self.verbose = verbose

	@abstractmethod
	def addon(self, Y_hat, p, *args, **kwargs):
		"""
		Post-process predictions.

		Parameters
		----------
		Y_hat : array-like, shape=(n_samples, n_classes)
			The raw prediction probabilities.

		p : array-like, shape=(n_samples,)
			The predictions to process. If no Addon
			processing was done prior to this, `p`
			corresponds with Y_hat.

		Returns
		-------
		p : array-like, shape=(n_samples,)
			The post-processed predictions.
		"""
		raise NotImplementedError("No addon function implemented")
