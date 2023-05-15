import numpy as np
from tqdm import trange

from sleepens.ml import cross_validate
from sleepens.ml._base_model import TimeSeriesClassifier
from sleepens.ml.models import TimeSeriesEnsemble

class StackedTimeSeriesEnsemble(TimeSeriesClassifier):
	"""
	A Stacked Time Series Ensemble is an Ensemble Classifier that
	stacks two Time Series Classifiers to better classify time series data.

	Parameters
	----------
	layer_1 : TimeSeriesClassifier, default=TimeSeriesEnsemble
		The first layer of TimeSeriesClassifier.

	layer_2 : TimeSeriesClassifier, default=TimeSeriesEnsemble
		The second layer of TimeSeriesClassifier.

	warm_start : bool, default=False
		Determines warm starting to allow training to pick
		up from previous training sessions.

	random_state : None or int or RandomState, default=None
		Initial seed for the RandomState. If `random_state` is None,
		return the RandomState singleton. If `random_state` is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	verbose : int, default=0
		Verbosity of estimator; higher values result in
		more verbose output.

	Attributes
	----------
	n_classes_ : int
		Number of classes.

	n_features_ : int
		Number of features.
	"""
	def __init__(self, layer_1=TimeSeriesEnsemble(), layer_2=TimeSeriesEnsemble(),
					warm_start=False, random_state=None, verbose=0):
		TimeSeriesClassifier.__init__(self, warm_start=warm_start,
							random_state=random_state, verbose=verbose)
		if not issubclass(type(layer_1), TimeSeriesClassifier):
			raise ValueError("Layer 1 must be a TimeSeriesClassifier")
		if not issubclass(type(layer_2), TimeSeriesClassifier):
			raise ValueError("Layer 2 must be a TimeSeriesClassifier")
		self.layer_1 = layer_1
		self.layer_2 = layer_2
		self.set_warm_start(warm_start)
		self.set_verbose(verbose)
		self.set_random_state(random_state)

	def _initialize(self):
		"""
		Initialize the parameters of the classifier.
		"""
		self.set_warm_start(self.warm_start)

	def fit(self, X, Y):
		"""
		Train the classifier on the given data and labels.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Training data.

		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		Returns
		-------
		self : Classifier
			Fitted classifier.
		"""
		X, Y = self._fit_setup(X, Y)
		if self.verbose > 1 : print("Fitting First Layer")
		score, X_2 = cross_validate(self.layer_1, X, Y, verbose=self.verbose-1)
		self.layer_1.fit(X, Y)
		if self.verbose > 1 : print("Fitting Second Layer")
		self.layer_2.fit(X_2, Y)
		if self.verbose > 1 : print("Completed training")
		return self

	def predict_proba(self, X):
		"""
		Return the prediction probabilities on the given data.

		Parameters
		----------
		X : list of ndarray, shape=(n_series, n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y_hat : list of ndarray, shape=(s_series, n_samples, n_classes)
			Probability predictions.
		"""
		X = self._predict_setup(X)
		if self.verbose > 1 : print("Predicting on First Layer")
		X_2 = self.layer_1.predict_proba(X)
		if self.verbose > 1 : print("Predicting on Second Layer")
		Y_hat = self.layer_2.predict_proba(X_2)
		if self.verbose > 1 : print("Completed predicting")
		return Y_hat

	def set_verbose(self, verbose):
		"""
		Set the verbosity of the Classifier.
		Subsidiary Classifiers have their verbosity
		suppressed by one compared to this Classifier.

		Parameters
		----------
		verbose : int, default=0
			Determines the verbosity of cross-validation.
			Higher verbose levels result in more output logged.
		"""
		TimeSeriesClassifier.set_verbose(self, verbose)
		for e in (self.layer_1, self.layer_2):
			e.set_verbose(verbose-1)

	def set_random_state(self, random_state):
		"""
		Set the RandomState of the Classifier.
		Subsidiary Classifiers set their RandomState
		based on this RandomState with a seed selected
		from 0 to 2**16.

		Parameters
		----------
		random_state : None or int or RandomState, default=None
			Initial seed for the RandomState. If `random_state` is None,
			return the RandomState singleton. If `random_state` is an int,
			return a RandomState with the seed set to the int.
			If `random_state` is a RandomState, return that RandomState.
		"""
		TimeSeriesClassifier.set_random_state(self, random_state)
		for e in (self.layer_1, self.layer_2):
			e.set_random_state(self.random_state.randint(0, 2**16))

	def set_warm_start(self, warm_start):
		"""
		Set the status of `warm_start` of the Classifier.

		Parameters
		----------
		warm_start : bool
			Determines warm starting to allow training to pick
			up from previous training sessions.
		"""
		TimeSeriesClassifier.set_warm_start(self, warm_start)
		for e in (self.layer_1, self.layer_2):
			e.set_warm_start(warm_start)

	def _is_fitted(self):
		"""
		Returns if the Classifier has been trained and is
		ready to predict new data.

		Returns
		-------
		fitted : bool
			True if the Classifier is fitted, False otherwise.
		"""
		attributes = ["n_classes_","n_features_"]
		return TimeSeriesClassifier._is_fitted(self, attributes=attributes) and \
				self.layer_1._is_fitted() and self.layer_2._is_fitted()
