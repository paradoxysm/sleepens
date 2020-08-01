from abc import ABC, abstractmethod

from sleepens.metanetwork.ml import get_loss
from sleepens.metanetwork.utils import check_XY

from sleepens.metanetwork.utils._estimator import Classifier
from sleepens.metanetwork.utils._base import Base

class EnsembleMember(Base, ABC):
	def __init__(self):
		pass

	@abstractmethod
	def forward(self, X):
		"""
		Conduct the forward propagation steps through the
		model.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Output.
		"""
		raise NotImplementedError("No forward function implemented")

	@abstractmethod
	def backward(self, dY):
		"""
		Conduct the backward propagation steps through the
		model.

		Parameters
		----------
		dY : array-like, shape=(n_samples, n_classes)
			Gradient of loss with respect to the output.

		Returns
		-------
		dY : array-like, shape=(n_samples, n_features)
			Gradient of loss with respect to the input.
		"""
		raise NotImplementedError("No backward function implemented")
