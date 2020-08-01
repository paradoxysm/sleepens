import numpy as np

from sleepens.ml import get_activation, get_regularizer
from sleepens.ml import get_optimizer
from sleepens.utils.misc import check_XY, create_random_state
from sleepens.analysis import get_metrics

from sleepens.utils._estimator import Classifier


class tsNeuralNetwork(Classifier):
	"""
	Time Series Neural Network takes in a
	time series of class probabilities and
	reclassifies the time series.

	The input and output layers are the same
	dimensions. Unlike a traditional Neural Network
	where the output layer is a softmax layer,
	Time Series Neural Networks' output layer
	consists of n softmax sub-layers, where n is
	the number of time-points.

	Parameters
	----------
	n : int, default=10
		The number of time points considered at a time.
		Must be a positive integer.

	layers : tuple, default=(100,)
		The ith element represents the number of
		neurons in the ith hidden layer.

	activation : str, Activation, default=PReLU(0.2)
		Activation function to use at each node in the NNDT.
		Must be one of the default loss functions or an
		object that extends Activation.

	loss : str, LossFunction, default='cross-entropy'
		Loss function to use for training. Must be
		one of the default loss functions or an object
		that extends LossFunction.

	optimizer : str, Optimizer, default='adam'
		Optimization method. Must be one of
		the default optimizers or an object that
		extends Optimizer.

	regularizer : str, Regularizer, None, default=None
		Regularization function to use at each node in the model.
		Must be one of the default regularizers or an object that
		extends Regularizer. If None, no regularization is done.

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
	n_layers : int
		Number of hidden/output layers in the model.

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

	weights_ : list of ndarray, shape=(layers,)
		Weights of the model.

	bias_ : list of ndarray, shape=(layers,)
		Biases of the model.
	"""
	def __init__(self, n=10, layers=(100,), activation='relu', loss='cross-entropy',
					optimizer='adam', regularizer=None, max_iter=100, tol=1e-4,
					batch_size=None, class_weight=None, metric='accuracy',
					warm_start=False, random_state=None, verbose=0):
		Classifier.__init__(self, loss=loss, max_iter=max_iter, tol=tol,
						batch_size=batch_size, class_weight=class_weight,
						metric=metric, warm_start=warm_start, verbose=verbose)
		self.n = n
		self.activation = get_activation(activation)
		self.optimizer = get_optimizer(optimizer)
		self.softmax = get_activation('softmax')
		self.regularizer = get_regularizer(regularizer)
		self.random_state = create_random_state(random_state)
		self.layers = layers
		self.output_layer = None
		self.n_layers_ = len(self.layers)
		self.weights_ = []
		self.bias_ = []
		self.output_weights_ = []
		self.output_bias_ = []
		self._x = []
		self._z = []
		self._output_x = []
		self._output_z = []

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
		self : NeuralNetwork
			Initialized model.
		"""
		super().initialize(n_features=n_features, n_classes=n_classes)
		if self.layers == tuple():
			for n in range(self.n):
				self.output_weights_.append(self.random_state.randn(self.n_features_, self.n_classes_) * 0.1)
				self.output_bias_.append(self.random_state.randn(self.n_classes_) * 0.1)
		else:
			self.weights_ = [self.random_state.randn(self.n_features_, self.layers[0]) * 0.1]
			self.bias_ = [self.random_state.randn(self.layers[0])]
			for l in range(self.n_layers_ - 1):
				self.weights_.append(self.random_state.randn(self.layers[l], self.layers[l+1]) * 0.1)
				self.bias_.append(self.random_state.randn(self.layers[l+1]))
			for n in range(self.n):
				self.output_weights_.append(self.random_state.randn(self.layers[-1], self.n_classes_) * 0.1)
				self.output_bias_.append(self.random_state.randn(self.n_classes_) * 0.1)
		keys = []
		for l in range(self.n_layers_):
			keys += ['w' + str(l), 'b' + str(l)]
		keys += ['woutput', 'boutput']
		self.optimizer.setup(keys)
		return self

	def forward(self, X):
		"""
		Conduct the forward propagation steps through the neural
		network.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Output.
		"""
		self._x, self._z, self._output_x, self._output_z = [], [], [], []
		for l in range(self.n_layers_):
			Z = np.dot(X, self.weights_[l]) + self.bias_[l]
			A = self.activation.activation(Z)
			self._z.append(Z)
			self._x.append(X)
			X = A
		for n in range(self.n):
			Z = np.dot(X, self.output_weights_[n]) + self.output_bias_[n]
			if n > 0:
				a = self.softmax.activation(Z)
				A[n:] += a[:-1]
				A.append(a[-1])
			else : A = self.softmax.activation(Z)
			self._output_z.append(Z)
			self._output_x = X
		A[1:-self.n-1] /= self.n
		return A

	def backward(self, dY):
		"""
		Conduct the backward propagation steps through the
		neural network.

		Parameters
		----------
		dY : array-like, shape=(n_samples, n_classes)
			Derivative of the output.

		Returns
		-------
		dY : array-like, shape=(n_samples, n_features)
			Derivative of the inputs.
		"""
		m = len(dY)
		for n in range(self.n):
			dZ = dY[n:-self.n-1-n] * self.softmax.gradiennt(self._output_z[n])
			dW = np.dot(self._output_x[n].T, dZ) / m
			db = np.sum(dZ, axis=0) / m
			if n > 0:
				dy = np.dot(dZ, self.output_weights_[n].T)
				dY_[n:] += dy[:-1]
				dY_.append(dy[-1])
			else : dY_ = np.dot(dZ, self.output_weights_[n].T)
		if self.regularizer is not None:
			dW += self.regularizer.gradient(self.output_weights_[n])
		self.output_weights_[n] -= self.optimizer.update('woutput', dW)
		self.output_bias_[n] -= self.optimizer.update('boutput', db)
		for l in range(self.n_layers_ - 1, -1, -1):
			if l == self.n_layers_ - 1:
				dZ = dY_ * self.softmax.gradient(self._z[-1])
			else : dZ = dY_ * self.activation.gradient(self._z[l])
			dW = np.dot(self._x[l].T, dZ) / m
			db = np.sum(dZ, axis=0) / m
			dY_ = np.dot(dZ, self.weights_[l].T)
			if self.regularizer is not None:
				dW += self.regularizer.gradient(self.weights_[l])
			self.weights_[l] -= self.optimizer.update('w' + str(l), dW)
			self.bias_[l] -= self.optimizer.update('b' + str(l), db)
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
		weights = len(self.weights_) > 0
		bias = len(self.bias_) > 0
		output_weights = len(self.output_weights_) > 0
		output_bias = len(self.output_bias_) > 0
		return weights and bias and output_weights and output_bias and self.fitted_
