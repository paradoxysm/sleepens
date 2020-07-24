import numpy as np
from abc import ABC, abstractmethod
from copy import copy
from tqdm import trange

from sleepens.ml import get_optimizer, get_loss
from sleepens.utils.misc import check_XY, one_hot, decode
from sleepens.utils.misc import calculate_batch, calculate_weight
from sleepens.io import BatchDataset
from sleepens.analysis import get_metrics

from sleepens.utils._base import Base


class BaseEstimator(Base, ABC):
	"""
	Base Estimator.

	Parameters
	----------
	metric : str, Metric, or None, default='accuracy'
		Metric for estimator score.

	verbose : int, default=0
		Verbosity of estimator; higher values result in
		more verbose output.

	Attributes
	----------
	n_classes_ : int
		Number of classes.

	n_features_ : int
		Number of features.

	fitted_ : bool
		True if the model has been deemed trained and
		ready to predict new data.
	"""
	def __init__(self, verbose=0, metric='accuracy'):
		if metric is None : metric = 'accuracy'
		self.metric = get_metrics(metric)
		self.verbose = verbose
		self.fitted_ = False
		self.n_classes_ = None
		self.n_features_ = None

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
		"""
		Predict class probabilities for each sample in `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		proba : array-like, shape=(n_samples, n_classes)
			Class probabilities of input data.
			The order of classes is in sorted ascending order.
		"""
		raise NotImplementedError("No predict_proba function implemented")

	def feature_importance(self, X, Y, score=None):
		"""
		Calculate the feature importances by permuting
		each feature separately and measuring the
		increase in loss.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Training data.

		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		score : Metric or None, default=None
			Score to compare feature importances.
			If None, uses the metric for the
			BaseEstimator.

		Returns
		-------
		importances : list, shape=(n_features,)
			List of feature importances by error increase,
			in order of features as they appear in the data.
			The larger the error increase, the more
			important the feature.
		"""
		if not self._is_fitted():
			raise RuntimeError("Model is not fitted")
		if score is None : score = self.metric
		elif not issubclass(score, Metric):
			raise ValueError("score must be a Metric or None")
		X, Y = check_XY(X=X, Y=Y)
		try : Y = one_hot(Y, cols=self.n_classes_)
		except : raise
		if X.shape[1] != self.n_features_:
			raise ValueError("Model takes %d features as input" % self.n_features_,
								"but data has %d features" % X.shape[1])
		if self.verbose > 0 : print("Calculating feature importances")
		loss = np.exp(score.score(self.predict_proba(X), Y))
		importances = []
		for f in range(X.shape[1]):
			X_ = copy(X)
			self.random_state.shuffle(X_[:,f])
			loss_ = score.score(self.predict_proba(X_), Y)
			importances.append(np.exp(loss_) / loss)
		return importances

	def score(self, Y, Y_hat, weights=None):
		"""
		Return mean metric of the estimator on the given
		data/predictions and target labels.

		If both data and predictions are provided, `score`
		just uses the predictions.

		Parameters
		----------
		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		Y_hat : array-like, shape=(n_samples,)
			Predicted labels.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights. If None, then samples are equally weighted.

		Returns
		-------
		score : float
			Mean metric score of the estimator for the given
			data/labels.
		"""
		if self.metric is not None:
			if weights is None : weights = np.ones(len(Y_hat))
			return self.metric.score(Y_hat, Y, weights=weights)
		return 0

	def _is_fitted(self):
		"""
		Return True if the model is properly ready
		for prediction.

		Returns
		-------
		fitted : bool
			True if the model can be used to predict data.
		"""
		return self.fitted_


class Classifier(BaseEstimator, ABC):
	"""
	Base Classifier using Backpropagation.

	Parameters
	----------
	loss : str, LossFunction, default='cross-entropy'
		Loss function to use for training. Must be
		one of the default loss functions or an object
		that extends LossFunction.

	optimizer : str, Optimizer, default='adam'
		Optimization method. Must be one of
		the default optimizers or an object that
		extends Optimizer.

	max_iter : int, default=10
		Maximum number of epochs to conduct during training.

	tol : float, default=1e-4
		Convergence criteria for early stopping.

	batch_size : int, float, default=None
		Batch size for training. Must be one of:
		 - int : Use `batch_size`.
		 - float : Use `batch_size * n_samples`.
		 - None : Use `n_samples`.

	class_weight : dict, 'balanced', or None, default=None
		Weights associated with classes in the form
		`{class_label: weight}`. Must be one of:
		 - None : All classes have a weight of one.
		 - 'balanced': Class weights are automatically calculated as
						`n_samples / (n_samples * np.bincount(Y))`.

	metric : str, Metric, or None, default='accuracy'
		Metric for estimator score.

	warm_start : bool, default=False
		Determines warm starting to allow training to pick
		up from previous training sessions.

	verbose : int, default=0
		Verbosity of estimator; higher values result in
		more verbose output.

	Attributes
	----------
	n_classes_ : int
		Number of classes.

	n_features_ : int
		Number of features.

	init_ : bool
		True if the model has been initialized
		and can predict data without failing.

	fitted_ : bool
		True if the model has been deemed trained and
		ready to predict new data.
	"""
	def __init__(self, loss='cross-entropy', max_iter=100, tol=1e-4,
					batch_size=None, class_weight=None, metric='accuracy',
					warm_start=False, verbose=0):
		BaseEstimator.__init__(self, verbose=verbose, metric=metric)
		self.loss = get_loss(loss)
		self.max_iter = max_iter
		self.tol = tol
		self.batch_size = batch_size
		self.class_weight = class_weight
		self.warm_start = warm_start
		self.init_ = False

	def initialize(self, n_features=None, n_classes=None):
		"""
		Initialize the model with random parameters.

		Parameters
		----------
		n_classes : int, default=None
			Number of classes. By default
			uses `n_classes_` set in model.

		n_features : int, default=None
			Number of features. By default
			uses `n_features_` set in model.

		Returns
		-------
		self : Classifier
			Initialized model.
		"""
		if self.warm_start and self.init_:
			return self
		if self.verbose > 0 : print("Initializing model")
		if n_features is not None : self.n_features_ = n_features
		if n_classes is not None : self.n_classes_ = n_classes
		if self.n_features_ is None or self.n_classes_ is None:
			raise ValueError("Features and Classes are not set")
		self.init_ = True
		return self

	def fit(self, X, Y, weights=None):
		"""
		Train the model on the given data and labels.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Training data.

		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights. If None, then samples are equally weighted.

		Returns
		-------
		self : Classifier
			Fitted estimator.
		"""
		X, Y = check_XY(X=X, Y=Y)
		try : Y = one_hot(Y, cols=self.n_classes_)
		except : raise
		if self.n_classes_ is None : self.n_classes_ = len(set(decode(Y)))
		if self.n_features_ is None : self.n_features_ = X.shape[1]
		weights = calculate_weight(decode(Y), self.n_classes_,
					class_weight=self.class_weight, weights=weights)
		batch_size = calculate_batch(self.batch_size, len(Y))
		ds = BatchDataset(X, Y, weights, seed=self.random_state).shuffle().repeat().batch(batch_size)
		self.initialize()
		if self.verbose > 0 : print("Training model for %d epochs" % self.max_iter,
								"on %d samples in batches of %d." % \
								(X.shape[0], batch_size))
		loss_prev, early_stop, e = np.inf, False, 0
		if self.verbose == 1 : epochs = trange(self.max_iter)
		else : epochs = range(self.max_iter)
		for e in epochs:
			batches = range(ds.n_batches)
			if self.verbose == 2 : batches = trange(ds.n_batches)
			elif self.verbose > 2 : print("Epoch %d" % e)
			for b in batches:
				X_batch, Y_batch, weights = ds.next()
				if len(X_batch) == 0:
					if self.verbose > 0 : print("No more data to train. Ending training.")
					early_stop = True
					break
				Y_hat = self.forward(X_batch)
				loss = np.mean(np.sum(self.loss.loss(Y_hat, Y_batch), axis=1))
				metric = self.score(Y_batch, Y_hat=Y_hat, weights=weights)
				msg = 'loss: %.4f' % loss + ', ' + self.metric.name + ': %.4f' % metric
				if self.verbose == 1 : epochs.set_description(msg)
				elif self.verbose == 2 : batches.set_description(msg)
				elif self.verbose > 2 : print("Epoch %d, Batch %d completed." % (e+1, b+1), msg)
				if self.tol is not None and np.abs(loss - loss_prev) < self.tol:
					early_stop = True
					break
				dY = self.loss.gradient(Y_hat, Y_batch) * weights.reshape(-1,1)
				self.backward(dY)
				loss_prev = loss
			if early_stop : break
		self.fitted_ = True
		if self.verbose > 0 : print("Training complete.")
		return self

	def predict_proba(self, X):
		"""
		Predict class probabilities for each sample in `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		proba : array-like, shape=(n_samples, n_classes)
			Class probabilities of input data.
			The order of classes is in sorted ascending order.
		"""
		if not self._is_fitted():
			raise RuntimeError("Model is not fitted")
		X = check_XY(X=X)
		if X.shape[1] != self.n_features_:
			raise ValueError("Model takes %d features as input" % self.n_features_,
								"but data has %d features" % X.shape[1])
		if self.verbose > 0 : print("Predicting %d samples." % \
								X.shape[0])
		return self.forward(X)

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
