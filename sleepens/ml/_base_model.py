"""Classifier and TimeSeriesClassifier"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from sklearn.utils.validation import check_is_fitted

from sleepens.analysis import get_metrics
from sleepens.utils import create_random_state, check_XY, one_hot
from sleepens.ml import get_loss

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
			Data to predict.

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
			Data to predict.

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
			Data to predict.

		Returns
		-------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Log probability predictions.
		"""
		return np.log(self.predict_proba(X))

	def feature_importance(self, X, Y, loss='mse', sort=True):
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

		loss : LossFunction, str
			LossFunction to look up. Must be one of:
			 - 'mse' : Mean Squared Error.
			 - 'mae' : Mean Absolute Error.
			 - 'huber' : Huber Loss.
			 - 'hinge' : Hinge Loss.
			 - 'cross-entropy' : Crossentropy Loss.
			 - LossFunction : A custom implementation.
			Custom LossFunctions must implement `loss`, `gradient`
			functions and contain `scale` attribute.

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
		if loss is None : raise ValueError("loss cannot be None")
		loss = get_loss(loss)
		X, Y = check_XY(X=X, Y=Y)
		if self.verbose > 1 : print("Calculating feature importances")
		Y_one_hot = one_hot(Y)
		p = self.predict_proba(X)
		error = np.exp(loss.loss(p, Y_one_hot))
		importances = []
		for f in range(X.shape[1]):
			X_p = deepcopy(X)
			self.random_state.shuffle(X_p[:,f])
			p_ = self.predict_proba(X_p)
			error_ = loss.loss(p_, Y_one_hot)
			importances.append((f, np.exp(error_) / error))
		if sort:
			return sorted(importances, key=lambda x: x[1])
		return importances

	def score(self, Y_hat, Y):
		"""
		Calculate the score of the given predictions compared
		to the target labels, as provided by the classifier's
		`metric`.

		Parameters
		----------
		Y_hat : array-like, shape=(n_samples,)
			Classifier predictions.

		Y : array-like, shape=(n_samples,)
			Target labels.

		Returns
		-------
		score : float, [0,1)
			The calculated score. If `metric` is None,
			returns 0.
		"""
		if self.metric is not None:
			return self.metric.score(Y_hat, Y)
		return 0

	def set_verbose(self, verbose):
		"""
		Set the verbosity of the Classifier.

		Parameters
		----------
		verbose : int, default=0
			Determines the verbosity of cross-validation.
			Higher verbose levels result in more output logged.
		"""
		self.verbose = verbose

	def set_random_state(self, random_state):
		"""
		Set the RandomState of the Classifier.

		Parameters
		----------
		random_state : None or int or RandomState, default=None
			Initial seed for the RandomState. If `random_state` is None,
			return the RandomState singleton. If `random_state` is an int,
			return a RandomState with the seed set to the int.
			If `random_state` is a RandomState, return that RandomState.
		"""
		self.random_state = create_random_state(random_state)

	def set_metric(self, metric):
		"""
		Set the metric of the Classifier.

		Parameters
		----------
		metric : Metric, None, str
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
		"""
		self.metric = get_metrics(metric)

	def set_warm_start(self, warm_start):
		"""
		Set the status of `warm_start` of the Classifier.

		Parameters
		----------
		warm_start : bool
			Determines warm starting to allow training to pick
			up from previous training sessions.
		"""
		self.warm_start = warm_start

	def _initialize(self):
		"""
		Initialize the parameters of the classifier.
		"""
		return

	def _fit_setup(self, X, Y):
		"""
		Set up the fit process.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Training data.

		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		Returns
		-------
		X : ndarray, shape=(n_samples, n_features)
			Training data.

		Y : ndarray, shape=(n_samples,)
			Target labels as integers.
		"""
		X, Y = check_XY(X=X, Y=Y)
		self._initialize()
		self._class_features(len(np.unique(Y)), X.shape[-1])
		if self.verbose > 1 : print("Training model")
		return X, Y

	def _predict_setup(self, X):
		"""
		Set up the predict process.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		X : ndarray, shape=(n_samples, n_features)
			Data to predict.
		"""
		if not self._is_fitted():
			raise RuntimeError("Model has not been fitted")
		X, _ = check_XY(X=X)
		if X.shape[1] != self.n_features_:
			raise ValueError("Model expects %d features" % self.n_features_,
								"but input has %d features" % X.shape[1])
		if self.verbose > 1 : print("Predicting %d samples." % X.shape[0])
		return X

	def _is_fitted(self, attributes=["n_classes_","n_features_"]):
		"""
		Returns if the Classifier has been trained and is
		ready to predict new data.

		Parameters
		----------
		attributes : list of string, default=["n_classes_","n_features_"]
			List of instance attributes to check existence of as proof
			that the Classifier has been fitted. Therefore, these attributes
			should only be created during the fit process.

		Returns
		-------
		fitted : bool
			True if the Classifier is fitted, False otherwise.
		"""
		try:
			check_is_fitted(self, attributes=attributes)
			return True
		except : return False

	def _class_features(self, n_classes, n_features):
		"""
		Set up the number of classes and features this
		model is designed to work with.

		If the Classifier has `warm_start` as True and
		already has `n_classes_` or `n_features_` set,
		the provided arguments must match.

		Parameters
		----------
		n_classes : int
			Number of classes.

		n_features : int
			Number of features.
		"""
		if not hasattr(self, 'n_classes_') or not self.warm_start : self.n_classes_ = n_classes
		elif self.warm_start and self.n_classes_ < n_classes:
			raise ValueError("Class mismatch: Model was trained on", self.n_classes_,
							"classes but input has", n_classes, "classes")
		if not hasattr(self, 'n_features_') or not self.warm_start : self.n_features_ = n_features
		elif self.warm_start and self.n_features_ != n_features:
			raise ValueError("Feature mismatch: Model has", self.n_features_,
							"features but input has", n_features, "features")

class TimeSeriesClassifier(Classifier):
	"""
	Base Time Series Classifier.

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
	def __init__(self, warm_start=False, metric='accuracy',
				random_state=None, verbose=0):
		Classifier.__init__(self, warm_start=warm_start, metric=metric,
							random_state=random_state, verbose=verbose)

	@abstractmethod
	def fit(self, X, Y):
		"""
		Train the classifier on the given data and labels.

		Parameters
		----------
		X : list of ndarray, shape=(n_series, n_samples, n_features)
			Training data.

		Y : list of ndarray, shape=(s_series, n_samples)
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
		X : list of ndarray, shape=(n_series, n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y_hat : list of ndarray, shape=(s_series, n_samples)
			Predictions.
		"""
		return [np.argmax(y, axis=1) for y in self.predict_proba(X)]

	@abstractmethod
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
		raise NotImplementedError("No predict_proba function implemented")

	def predict_log_proba(self, X):
		"""
		Return the prediction log probabilities on the given data.

		Parameters
		----------
		X : list of ndarray, shape=(n_series, n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y_hat : list of ndarray, shape=(s_series, n_samples, n_classes)
			Log probability predictions.
		"""
		return [np.log(y) for y in self.predict_proba(X)]

	def feature_importance(self, X, Y, loss='mse', sort=True):
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
		X : list of ndarray, shape=(n_series, n_samples, n_features)
			Data.

		Y : list of ndarray, shape=(n_series, n_samples,)
			Target labels as integers.

		loss : LossFunction, str
			LossFunction to look up. Must be one of:
			 - 'mse' : Mean Squared Error.
			 - 'mae' : Mean Absolute Error.
			 - 'huber' : Huber Loss.
			 - 'hinge' : Hinge Loss.
			 - 'cross-entropy' : Crossentropy Loss.
			 - LossFunction : A custom implementation.
			Custom LossFunctions must implement `loss`, `gradient`
			functions and contain `scale` attribute.

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
		if loss is None : raise ValueError("loss cannot be None")
		loss = get_loss(loss)
		for x, y in zip(X, Y) : check_XY(X=x, Y=y)
		try : X_, Y_ = np.concatenate(X), np.concatenate(Y)
		except : raise ValueError("Inputs have different number of features")
		if self.verbose > 1 : print("Calculating feature importances")
		Y_one_hot = np.concatenate([one_hot(y) for y in Y])
		p = np.concatenate(self.predict_proba(X))
		error = np.mean(np.exp(loss.loss(p, Y_one_hot)))
		importances = []
		for f in range(X_.shape[1]):
			if self.verbose > 2 : print("Permuting feature", f+1)
			X_p = [deepcopy(x) for x in X]
			for x in X_p : self.random_state.shuffle(x[:,f])
			p_ = np.concatenate(self.predict_proba(X_p))
			error_ = np.mean(np.exp(loss.loss(p_, Y_one_hot)))
			importances.append((f, error_ / error))
		if sort:
			return sorted(importances, key=lambda x: x[1])
		return importances

	def score(self, Y_hat, Y):
		"""
		Calculate the score of the given predictions compared
		to the target labels, as provided by the classifier's
		`metric`.

		Parameters
		----------
		Y_hat : list of ndarray, shape=(n_series, n_samples,)
			Classifier predictions.

		Y : list of ndarray, shape=(n_series, n_samples,)
			Target labels.

		Returns
		-------
		score : float, [0,1)
			The calculated score. If `metric` is None,
			returns 0.
		"""
		if self.metric is not None:
			scores = [self.metric.score(y_hat, y) for y_hat, y in zip(Y_hat, Y)]
			lengths = [len(y_hat) for y_hat in Y_hat]
			return np.average(scores, weights=lengths)
		return 0

	def _fit_setup(self, X, Y):
		"""
		Set up the fit process.

		Parameters
		----------
		X : list of ndarray, shape=(n_series, n_samples, n_features)
			Training data.

		Y : list of ndarray, shape=(s_series, n_samples)
			Target labels as integers.

		Returns
		-------
		X : list of ndarray, shape=(n_series, n_samples, n_features)
			Training data.

		Y : list of ndarray, shape=(s_series, n_samples)
			Target labels as integers.
		"""
		for x, y in zip(X, Y) : check_XY(X=x, Y=y)
		try : X_, Y_ = np.concatenate(X), np.concatenate(Y)
		except : raise ValueError("Inputs have different number of features")
		self._initialize()
		self._class_features(len(np.unique(Y_)), X_.shape[-1])
		if self.verbose > 1 : print("Training model")
		return X, Y

	def _predict_setup(self, X):
		"""
		Set up the predict process.

		Parameters
		----------
		X : list of ndarray, shape=(n_series, n_samples, n_features)
			Data to predict.

		Returns
		-------
		X : list of ndarray, shape=(n_series, n_samples, n_features)
			Data to predict.
		"""
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
