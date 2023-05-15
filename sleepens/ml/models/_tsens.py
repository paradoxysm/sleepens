import numpy as np
from tqdm import trange
from copy import deepcopy

from sleepens.ml import check_estimator
from sleepens.ml._base_model import TimeSeriesClassifier
from sleepens.ml.models import GradientBoostingClassifier
from sleepens.utils import time_array

default_estimators = [GradientBoostingClassifier() for i in range(10)]

class TimeSeriesEnsemble(TimeSeriesClassifier):
	"""
	A Time Series Ensemble is an Ensemble Classifier that
	uses an array of estimators to classify time series data
	with a rolling window.

	Each ensemble member views the same window of the time series
	but predicts at a given timepoint within the window. All predictions
	from visited ensemble members are then averaged to produce the overall
	prediction.

	The Time Series Ensemble provides two main advantages:
	1. A rolling window style of classification that does not truncate
		data on either end.
	2. A rolling window that is tapers to twice the size of the given window.
	Finally, the Time Series Ensemble is implemented as a bi-directional
	Classifier, evaluating with both past and future timepoints.

	Parameters
	----------
	estimators : list of Classifier, default=[10*GradientBoostingClassifier]
		The ensemble array. The number of classifiers is assumed
		as the window size. The i-th Classifier will predict for
		the i-th timepoint within the moving window.

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

	n_features_in_ : int
		Number of features.
	"""
	def __init__(self, estimators=default_estimators,
					warm_start=False, random_state=None, verbose=0):
		TimeSeriesClassifier.__init__(self, warm_start=warm_start,
							random_state=random_state, verbose=verbose)
		for e in estimators:
			check_estimator(e)
		self.n = len(estimators)
		self.base_estimators = estimators
		self.set_warm_start(warm_start)
		self.set_verbose(verbose)
		self.set_random_state(random_state)

	def _initialize(self):
		"""
		Initialize the parameters of the classifier.
		"""
		if self._is_fitted():
			for e in self.estimators_ : e.set_warm_start(self.warm_start)
		else:
			self.estimators_ = deepcopy(self.base_estimators)

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
		X_a, Y_a = [time_array(x, self.n) for x in X], [time_array(y, self.n) for y in Y]
		X_a, Y_a = np.concatenate(X_a), np.concatenate(Y_a)
		if self.verbose == 1 or self.verbose == 2 : stages = trange(self.n)
		else : stages = range(self.n)
		for s in stages:
			if self.verbose > 2 : print("Fitting estimator", s)
			self.estimators_[s].fit(X_a, Y_a[:,s])
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
		X_a = [time_array(x, self.n) for x in X]
		Y_hat = [np.zeros((len(x), self.n_classes_)) for x in X]
		count = [np.zeros((len(x),), dtype=int) for x in X]
		if self.verbose == 1 or self.verbose == 2 : stages = trange(self.n)
		else : stages = range(self.n)
		for s in stages:
			if self.verbose > 2 : print("Predicting on estimator", s)
			for i in range(len(X)):
				Y_hat[i][s:s+len(X_a[i])] += self.estimators_[s].predict_proba(X_a[i])
				count[i][s:s+len(X_a[i])] += 1
		for i in range(len(Y_hat)) : Y_hat[i] /= count[i].reshape(-1,1)
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
		estimators = self.estimators_ if hasattr(self, 'estimators_') else self.base_estimators
		for e in estimators:
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
		estimators = self.estimators_ if hasattr(self, 'estimators_') else self.base_estimators
		for e in estimators:
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
		estimators = self.estimators_ if hasattr(self, 'estimators_') else self.base_estimators
		for e in estimators:
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
		attributes = ["estimators_","n_classes_","n_features_in_"]
		return TimeSeriesClassifier._is_fitted(self, attributes=attributes)
