from abc import ABC, abstractmethod

from sleepens.utils.misc import create_random_state
from sleepens.utils._base import Base

class AbstractSleepEnsemble(Base, ABC):
	def __init__(self, random_state=None, verbose=0):
		self.random_state = create_random_state(seed=random_state)
		self.verbose = verbose

	@abstractmethod
	def fit(self, X, Y, weight=None):
		raise NotImplementedError("No fit function implemented")

	def predict(self, X):
		"""
		Predict classes for each sample in `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y : array-like, shape=(n_samples,)
			Predicted labels.
		"""
		pred = self.predict_proba(X)
		pred = np.argmax(pred, axis=1)
		return pred

	def predict_log_proba(self, X):
		"""
		Predict class log-probabilities for each sample in `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		proba : array-like, shape=(n_samples, n_classes)
			Class log-probabilities of input data.
			The order of classes is in sorted ascending order.
		"""
		return np.log(self.predict_proba(X))

	@abstractmethod
	def predict_proba(self, X):
		raise NotImplementedError("No predict_proba function implemented")

	@abstractmethod
	def process(self, filepath, labels=False):
		raise NotImplementedError("No process function implemented")

	@abstractmethod
	def export_model(self):
		raise NotImplementedError("No export_model function implemented")

	@abstractmethod
	def load_model(self, filepath):
		raise NotImplementedError("No load_model function implemented")
