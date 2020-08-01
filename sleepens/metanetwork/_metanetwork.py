import numpy as np

from sleepens.metanetwork.utils import check_XY, create_random_state

from sleepens.metanetwork import EnsembleMember
from sleepens.metanetwork.utils._estimator import Classifier


class MetaNetwork(Classifier, EnsembleMember):
	"""
	Meta-Network.

	Parameters
	----------
	network : array-like, shape=(n_layers, n_estimators)
		List of ensemble members in an array formatted
		to the structure of the meta-network.

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
	network : list, shape=(n_layers, n_estimators)
		Network of estimators.

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
	def __init__(self, network, loss='cross-entropy', max_iter=100, tol=1e-4,
					batch_size=None, class_weight=None, warm_start=False,
					metric='accuracy', verbose=0, random_state=None):
		Classifier.__init__(self, loss=loss, max_iter=max_iter, tol=tol,
						batch_size=batch_size, class_weight=class_weight,
						metric=metric, warm_start=warm_start, verbose=verbose)
		self.network = network
		self.random_state = create_random_state(random_state)

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
		self : MetaNetwork
			Initialized model.
		"""
		super().initialize(n_features=n_features, n_classes=n_classes)
		c = 0
		for layer in self.network:
			f = self.n_features_ + c
			c = 0
			for estimator in layer:
				estimator.initialize(n_features=f)
				c += estimator.n_classes_
		self._validate_network()
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
		for layer in self.network:
			for estimator in layer:
				estimator.fitted_ = True
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
		X_ = X
		for layer in self.network:
			Y_hat = np.array([]).reshape(len(X), -1)
			for estimator in layer:
				Y_hat = np.concatenate((Y_hat, estimator.forward(X_)), axis=1)
			X_ = np.concatenate((X, Y_hat), axis=1)
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
		for l in range(len(self.network)-1, -1, -1):
			i = 0
			dY_ = None
			for estimator in self.network[l]:
				n_out = i + estimator.n_classes_
				dY_e = estimator.backward(dY[:, i:n_out])
				if dY_ is None : dY_ = dY_e
				else : dY_ += dY_e
				i = n_out
			if l > 0 : dY = dY_[:, self.n_features_:]
		else : dY = dY_
		return dY

	def _is_fitted(self):
		"""
		Return True if the model is properly ready
		for prediction.

		Returns
		-------
		fitted : bool
			True if the model can be used to predict data.
		"""
		network = True
		for layer in self.network:
			for estimator in layer:
				network = network and estimator._is_fitted()
		return network and self.fitted_

	def _validate_network(self):
		"""
		Validate the network.

		Parameters
		----------
		network : list, shape=(n_layers, n_estimators)
			Arrangement of base learners in the MetaNetwork.
			Base learners need to have `n_features_` and
			`n_classes_` predetermined.
		"""
		for layer in self.network:
			for estimator in layer:
				if not isinstance(estimator, EnsembleMember):
					raise ValueError("All estimators must extend EnsembleMember")
				elif estimator.n_classes_ is None:
					raise ValueError("All estimators must have n_classes_ set")
		if len(self.network[-1]) > 1:
			n_classes = np.sum([e.n_classes_ for e in self.network[-1]])
		else : n_classes = self.network[-1][0].n_classes_
		if n_classes != self.n_classes_:
			raise ValueError("Estimator(s) in the last layer need to output %d class" % self.n_classes_)
