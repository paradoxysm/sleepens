"""Classifier and TimeSeriesClassifier"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

from abc import ABC, abstractmethod
import numpy as np
from sklearn.utils.validation import check_is_fitted

from sleepens.analysis import get_metrics
from sleepens.utils import create_random_state, check_XY

def check_estimator(estimator):
	"""
	Checks that `estimator` is a subclass of
	Classifier. Raises an error if
	it is not.

	Parameters
	----------
	estimator : object
		Object to check.
	"""
	if not issubclass(type(estimator), Classifier):
		raise ValueError("Object is not a Classifier")

def check_timeseries_estimator(estimator):
	"""
	Checks that `estimator` is a subclass of
	TimeSeriesClassifier. Raises an error if
	it is not.

	Parameters
	----------
	estimator : object
		Object to check.
	"""
	if not issubclass(type(estimator), TimeSeriesClassifier):
		raise ValueError("Object is not a TimeSeriesClassifier")

class Classifier(ABC):
	"""
	Base Classifier.

	Parameters
	----------
	warm_start : bool, default=False
		Determines warm starting to allow training to pick
		up from previous training sessions.

	metric : Metric, None, str, default='accuracy'
		Metric to look up. Must be one of:
		 - 'accuracy' : Accuracy.
		 - 'precision' : Precision.
		 - 'recall' : Recall.
		 - 'f-score' : F1-Score.
		 - 'roc-auc' : ROC-AUC.
		 - Metric : A custom implementation.
		 - None : Return None.
		Custom Metrics must implement `score` which
		by default should return a single float value.

	random_state : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
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
	def __init__(self, warm_start=False, metric='accuracy',
				random_state=None, verbose=0):
		self.warm_start = warm_start
		if metric is None : metric = 'accuracy'
		self.metric = get_metrics(metric)
		self.random_state = create_random_state(random_state)
		self.verbose = verbose

	@abstractmethod
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
		raise NotImplementedError("No fit function implemented")

	def predict(self, X):
		"""
		Predict on the given data.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict

		Returns
		-------
		Y_hat : array-like, shape=(n_samples,)
			Predictions.
		"""
		return np.argmax(self.predict_proba(X), axis=1)

	@abstractmethod
	def predict_proba(self, X):
		"""
		Return the prediction probabilities on the given data.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict

		Returns
		-------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Probability predictions.
		"""
		raise NotImplementedError("No predict_proba function implemented")

	def predict_log_proba(self, X):
		"""
		Return the prediction log probabilities on the given data.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict

		Returns
		-------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Log probability predictions.
		"""
		return np.log(self.predict_proba(X))

	def feature_importance(self, X, Y, metric=None, sort=True):
		"""
		Return the importances of features through Permutation
		Feature Importance (PFI). In PFI, the fitted classifier
		predicts over data where one feature has been permuted randomly.
		The resulting change in score provides a measure of that feature's
		importance.

		Take care when working with data that contains correlated features.
		The results of PFI may not represent true feature importances in such cases,
		due to either permutations resulting in impossible data or a reduction in
		apparent feature importance due to the sharing of utility among correlated features.

		If conducted with the training data, PFI shows the most salient features
		used by the classifier. If conducted with unseen data, PFI describes
		the most valuable features for predictions.

		Remember that PFI relates to the specific classifier in question.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data.

		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		metric : Metric, None, str, default=None
			Metric to look up. Must be one of:
			 - 'accuracy' : Accuracy.
			 - 'precision' : Precision.
			 - 'recall' : Recall.
			 - 'f-score' : F1-Score.
			 - 'roc-auc' : ROC-AUC.
			 - Metric : A custom implementation.
			 - None : Use this Classifier's metric.
			Custom Metrics must implement `score` which
			by default should return a single float value.

		sort : bool, default=True
			Determine if the feature importances should be sorted.

		Returns
		-------
		importances : list of tuple (feature, importance)
			List of each feature and the corresponding importance.
			The feature is labelled numerically in the order they appear
			in the data. Importance is the change in the score
			from permuting that feature.
		"""
		if not self._is_fitted():
			raise RuntimeError("Model has not been initialized")
		if metric is None : metric = self.metric
		elif not issubclass(metric, Metric):
			raise ValueError("score must be a Metric or None")
		X, Y = check_XY(X=X, Y=Y)
		if X.shape[1] != self.n_features_:
			raise ValueError("Model expects %d features" % self.n_features_,
								"but input has %d features" % X.shape[1])
		if self.verbose > 1 : print("Calculating feature importances")
		loss = np.exp(self.score(self.predict_proba(X), Y))
		importances = []
		for f in range(X.shape[1]):
			X_ = copy(X)
			self.random_state.shuffle(X_[:,f])
			loss_ = score.score(self.predict_proba(X_), Y)
			importances.append((f, np.exp(loss_) / loss))
		if sort:
			return sorted(importances, key=lambda x: x[1])
		return importances

	def score(self, Y_hat, Y):
		if self.metric is not None:
			return self.metric.score(Y_hat, Y)
		return 0

	def set_verbose(self, verbose):
		self.verbose = verbose

	def set_random_state(self, random_state):
		self.random_state = create_random_state(random_state)

	def set_metric(self, metric):
		self.metric = get_metrics(metric)

	def set_warm_start(self, warm_start):
		self.warm_start = warm_start

	def _initialize(self):
		return

	def _fit_setup(self, X, Y):
		X, Y = check_XY(X=X, Y=Y)
		self._initialize()
		self._class_features(len(np.unique(Y)), X.shape[-1])
		if self.verbose > 1 : print("Training model")
		return X, Y

	def _predict_setup(self, X):
		if not self._is_fitted():
			raise RuntimeError("Model has not been fitted")
		X, _ = check_XY(X=X)
		if X.shape[1] != self.n_features_:
			raise ValueError("Model expects %d features" % self.n_features_,
								"but input has %d features" % X.shape[1])
		if self.verbose > 1 : print("Predicting %d samples." % X.shape[0])
		return X

	def _is_fitted(self, attributes=["n_classes_","n_features_"]):
		try:
			check_is_fitted(self, attributes=attributes)
			return True
		except : return False

	def _class_features(self, n_classes, n_features):
		if not hasattr(self, 'n_classes_') or not self.warm_start : self.n_classes_ = n_classes
		elif self.warm_start and self.n_classes_ < n_classes:
			raise ValueError("Class mismatch: Model was trained on", self.n_classes_,
							"classes but input has", n_classes, "classes")
		if not hasattr(self, 'n_features_') or not self.warm_start : self.n_features_ = n_features
		elif self.warm_start and self.n_features_ != n_features:
			raise ValueError("Feature mismatch: Model has", self.n_features_,
							"features but input has", n_features, "features")

class TimeSeriesClassifier(Classifier):
	def __init__(self, warm_start=False, metric='accuracy',
				random_state=None, verbose=0):
		Classifier.__init__(self, warm_start=warm_start, metric=metric,
							random_state=random_state, verbose=verbose)

	@abstractmethod
	def fit(self, X, Y):
		raise NotImplementedError("No fit function implemented")

	def predict(self, X):
		return [np.argmax(y, axis=1) for y in self.predict_proba(X)]

	@abstractmethod
	def predict_proba(self, X):
		raise NotImplementedError("No predict_proba function implemented")

	def predict_log_proba(self, X):
		return [np.log(y) for y in self.predict_proba(X)]

	def _fit_setup(self, X, Y):
		for x, y in zip(X, Y) : check_XY(X=x, Y=y)
		try : X_, Y_ = np.concatenate(X), np.concatenate(Y)
		except : raise ValueError("Inputs have different number of features")
		self._initialize()
		self._class_features(len(np.unique(Y_)), X_.shape[-1])
		if self.verbose > 1 : print("Training model")
		return X, Y

	def _predict_setup(self, X):
		if not self._is_fitted():
			raise RuntimeError("Model has not been fitted")
		for x in X : check_XY(X=x)
		try : X_ = np.concatenate(X)
		except : raise ValueError("Inputs have different number of features")
		if X_.shape[1] != self.n_features_:
			raise ValueError("Model expects %d features" % self.n_features_,
								"but input has %d features" % X_.shape[1])
		if self.verbose > 1 : print("Predicting %d samples." % X_.shape[0])
		return X
