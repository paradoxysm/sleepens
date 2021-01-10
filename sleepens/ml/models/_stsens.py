import numpy as np
from tqdm import trange

from sleepens.analysis import get_metrics
from sleepens.ml import cross_validate
from sleepens.ml._base_model import TimeSeriesClassifier
from sleepens.ml.models import TimeSeriesEnsemble

class StackedTimeSeriesEnsemble(TimeSeriesClassifier):
	def __init__(self, layer_1=TimeSeriesEnsemble(), layer_2=TimeSeriesEnsemble(),
					warm_start=False, metric='accuracy', random_state=None, verbose=0):
		TimeSeriesClassifier.__init__(self, warm_start=warm_start, metric=metric,
							random_state=random_state, verbose=verbose)
		if not issubclass(type(layer_1), TimeSeriesClassifier):
			raise ValueError("Layer 1 must be a TimeSeriesClassifier")
		if not issubclass(type(layer_2), TimeSeriesClassifier):
			raise ValueError("Layer 2 must be a TimeSeriesClassifier")
		self.layer_1 = layer_1
		self.layer_2 = layer_2
		self.set_warm_start(warm_start)
		self.set_metric(metric)
		self.set_verbose(verbose)
		self.set_random_state(random_state)

	def fit(self, X, Y):
		X, Y = self._fit_setup(X, Y)
		if self.verbose > 1 : print("Fitting First Layer")
		score, X_2 = cross_validate(self.layer_1, X, Y, verbose=self.verbose-1)
		self.layer_1.fit(X, Y)
		if self.verbose > 1 : print("Fitting Second Layer")
		self.layer_2.fit(X_2, Y)
		if self.verbose > 1 : print("Completed training")
		return self

	def predict_proba(self, X):
		X = self._predict_setup(X)
		if self.verbose > 1 : print("Predicting on First Layer")
		X_2 = self.layer_1.predict_proba(X)
		if self.verbose > 1 : print("Predicting on Second Layer")
		Y_hat = self.layer_2.predict_proba(X_2)
		if self.verbose > 1 : print("Completed predicting")
		return Y_hat

	def set_verbose(self, verbose):
		TimeSeriesClassifier.set_verbose(self, verbose)
		for e in (self.layer_1, self.layer_2):
			e.set_verbose(verbose-1)

	def set_random_state(self, random_state):
		TimeSeriesClassifier.set_random_state(self, random_state)
		for e in (self.layer_1, self.layer_2):
			e.set_random_state(self.random_state.randint(0, 2**16))

	def set_metric(self, metric):
		TimeSeriesClassifier.set_metric(self, metric)
		for e in (self.layer_1, self.layer_2):
			e.set_metric(metric)

	def set_warm_start(self, warm_start):
		TimeSeriesClassifier.set_warm_start(self, warm_start)
		for e in (self.layer_1, self.layer_2):
			e.set_warm_start(warm_start)

	def _is_fitted(self):
		attributes = ["n_classes_","n_features_"]
		return TimeSeriesClassifier._is_fitted(self, attributes=attributes) and \
				self.layer_1._is_fitted() and self.layer_2._is_fitted()
