import numpy as np
from itertools import combinations
from copy import deepcopy

from sleepens.metanetwork.ml import get_loss
from sleepens.metanetwork.utils import check_XY, create_random_state

from sleepens.metanetwork import NeuralNetwork, EnsembleMember
from sleepens.metanetwork.utils._estimator import Classifier


class ArbitratedRecallNetwork(Classifier, EnsembleMember):
	"""
	An Arbitrated Network is a specific structural
	form of meta-network.

	Parameters
	----------
	estimator : EnsembleMember, default=NeuralNetwork()
		The base estimator type to use for the construction
		of ensemble members.

	loss : str, LossFunction, default='cross-entropy'
		Loss function to use for training. Must be
		one of the default loss functions or an object
		that extends LossFunction.

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
	first_ : list of EnsembleMember
		First layer ensemble members.

	second_ : list of EnsembleMember
		Second layer of ensemble members.

	catch_ : EnsembleMember
		Catching ensemble member.

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
	def __init__(self, estimator=NeuralNetwork(), loss='cross-entropy',
					max_iter=100, tol=1e-4,
					batch_size=None, class_weight=None, warm_start=False,
					metric='accuracy', verbose=0, random_state=None):
		Classifier.__init__(self, loss=loss, max_iter=max_iter, tol=tol,
						batch_size=batch_size, class_weight=class_weight,
						metric=metric, warm_start=warm_start, verbose=verbose)
		self.estimator = self._validate_estimator(estimator)
		self.random_state = create_random_state(random_state)
		self.first_ = []
		self.second_ = []
		self.catch_ = None
		self._order = []
		self._s = []
		self._c = []

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
		self : ArbitratedNetwork
			Initialized model.
		"""
		super().initialize(n_features=n_features, n_classes=n_classes)
		self.first_, self.second_ = [], []
		for c in range(self.n_classes_):
			seed = int(self.random_state.rand(1) * 1000)
			e = deepcopy(self.estimator)
			e.random_state = create_random_state(seed=seed)
			e.initialize(n_features=self.n_features_, n_classes=2)
			self.first_.append(e)
		n_second = int(self.n_classes_ * (self.n_classes_ - 1) / 2)
		for a in range(n_second):
			seed = int(self.random_state.rand(1) * 1000)
			e = deepcopy(self.estimator)
			e.random_state = create_random_state(seed=seed)
			n = 2 if self.n_classes_ == 2 else 3
			e.initialize(n_features=self.n_features_ + self.n_classes_, n_classes=n)
			self.second_.append(e)
		seed = int(self.random_state.rand(1) * 1000)
		self.catch_ = deepcopy(self.estimator)
		self.catch_.random_state = create_random_state(seed=seed)
		self.catch_.initialize(n_features=self.n_features_ + self.n_classes_,
		 						n_classes=self.n_classes_)
		self._order = list(combinations(np.arange(self.n_classes_), 2))
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
		super().fit(X, Y, weights=None)
		for estimator in self.first_:
			estimator.fitted_ = True
		for estimator in self.second_:
			estimator.fitted_ = True
		self.catch_.fitted_ = True
		return self

	def forward(self, X):
		"""
		Conduct the forward propagation steps through the
		meta-network.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Output.
		"""
		# Pass through base OVRs
		f_p, f_s = np.array([]).reshape(len(X), -1), np.array([]).reshape(len(X), -1)
		for e in self.first_:
			p = e.forward(X)
			# Collect only the positive class
			f_p = np.concatenate((f_p, p[:,1].reshape(-1,1)), axis=1)
			# Collect signal result
			f_s = np.concatenate((f_s, np.argmax(p, axis=1).reshape(-1,1)), axis=1)
		# Pass through arbitrators
		s_p, s_s = np.zeros_like(f_p), np.array([]).reshape(len(X), -1).astype(bool)
		c_s = deepcopy(f_s)
		X_ = np.concatenate((X, f_p), axis=1)
		for i in range(len(self._order)):
			o = self._order[i]
			# Identify arbitrator activation
			signal = np.maximum(0, np.sum(f_s[:,o], axis=1)-1).astype(bool)
			if True in signal:
				p_ = self.second_[i].forward(X_[signal])
				p = np.zeros((len(p_), self.n_classes_))
				if self.n_classes_ > 2:
					# Split the negative class by the constituent OVR results
					p[:,o] = p_[:,:-1]
					neg = list(set(np.arange(self.n_classes_)) - set(o))
					neg_f = f_p[signal]
					neg_f = neg_f[:,neg]
					p[:,neg] = neg_f * p_[:,-1].reshape(-1,1)
				else:
					p = p_
				# Aggregate arbitrator results
				s_p[signal] += p
				# Calculate arbitration results
				s = np.zeros(len(X_))
				s[signal] = np.argmax(p_, axis=1)
				pos_a, pos_b, pos_n = (np.where(s == 0)[0], np.where(s == 1)[0],
										np.where(s == 2)[0])
				if len(pos_a) > 0 : c_s[pos_a, o[0]] += 1
				if len(pos_b) > 0 : c_s[pos_b, o[1]] += 1
				if len(pos_n) > 0 : c_s[pos_n.reshape(-1,1), o] -= 1
			s_s = np.concatenate((s_s, signal.reshape(-1,1)), axis=1)
		# Scale to 1
		s_p_sum = np.maximum(1e-8, np.sum(s_p, axis=1)).reshape(-1,1)
		s_p /= s_p_sum
		# Pass through catcher
		c_s_test = c_s.max(axis=1).reshape(-1,1).astype(int) - 1
		c_s_test = np.sum(np.maximum(0, c_s - c_s_test), axis=1)
		signal = np.where(c_s_test > 1)[0]
		p = self.catch_.forward(X_[signal])
		# Combine all results
		Y_hat = np.zeros_like(f_p)
		f_p_sum = np.maximum(1e-8, np.sum(f_p, axis=1)).reshape(-1,1)
		Y_hat = f_p / f_p_sum
		mask = np.where(np.sum(s_s, axis=1) > 0)[0]
		Y_hat[mask] = s_p[mask]
		Y_hat[signal] = p
		self._s = s_s
		self._c = signal
		return Y_hat

	def backward(self, dY):
		"""
		Conduct the backward propagation steps through the
		meta-network.

		Parameters
		----------
		dY : array-like, shape=(n_samples, n_classes)
			Derivative of the output.

		Returns
		-------
		dY : array-like, shape=(n_samples, n_features)
			Derivative of the inputs.
		"""
		dY_ = deepcopy(dY)
		# Identify catcher activation
		signal = self._c
		if True in signal:
			dY_[signal] = self.catch_.backward(dY[signal])[:, self.n_features_:]
		# Identify arbitrator activation
		mask = np.where(np.sum(self._s, axis=1) > 0)[0]
		dY_[mask] *= 0
		for i in range(len(self._order)):
			o = self._order[i]
			signal = self._s[:, i]
			if True not in signal : continue
			dY_i = dY[signal]
			dY_i = dY_i[:,o]
			if self.n_classes_ > 2:
				neg = list(set(np.arange(self.n_classes_)) - set(o))
				dneg = dY[signal]
				dneg = np.sum(dneg[:,neg].reshape(-1,len(neg)), axis=1).reshape(-1,1)
				dY_i = np.concatenate((dY_i, dneg), axis=1)
			dY_[signal] += self.second_[i].backward(dY_i)[:, self.n_features_:]
		dY_ = np.zeros((len(dY), self.n_features_))
		recall_mask = [1, 0.05, 1, 0.05]
		for i in range(len(self.first_)):
			dY_i = dY[:,i].reshape(-1,1)
			# Option1: Mask only what is positive class - only learns this...
			# Option2: Recall skew hyperparameter
			mask = np.where(dY_i < 0)
			dY_i[mask] *= recall_mask[i]
			dY_i = np.concatenate((-1 * dY_i, dY_i), axis=1)
			dY_ += self.first_[i].backward(dY_i)
		return dY_

	def _is_fitted(self):
		"""
		Return True if the model is properly ready
		for prediction.

		Returns
		-------
		fitted : bool
			True if the model can be used to predict data.
		"""
		first = [f._is_fitted() for f in self.first_]
		first = np.all(first) and len(first) > 0
		second = [s._is_fitted() for s in self.second_]
		second = np.all(second) and len(second) > 0
		catch = self.catch_ is None or self.catch_._is_fitted()
		return first and second and catch and self.fitted_

	def _validate_estimator(self, estimator):
		"""
		Validate `estimator` as an EnsembleMember.

		Parameters
		----------
		estimator : object
			Estimator to validate.

		Returns
		-------
		estimator : EnsembleMember
			Estimator.
		"""
		if not issubclass(type(estimator), EnsembleMember):
			raise ValueError("Estimator must inherit EnsembleMember")
		return estimator
