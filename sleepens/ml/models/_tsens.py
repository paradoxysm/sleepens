import numpy as np
from tqdm import trange
from copy import deepcopy

from sleepens.ml import check_estimator
from sleepens.ml._base_model import TimeSeriesClassifier
from sleepens.ml.models import GradientBoostingClassifier
from sleepens.utils import time_array

default_estimators = [GradientBoostingClassifier() for i in range(10)]

class TimeSeriesEnsemble(TimeSeriesClassifier):
	def __init__(self, estimators=default_estimators,
					warm_start=False, metric='accuracy', random_state=None, verbose=0):
		TimeSeriesClassifier.__init__(self, warm_start=warm_start, metric=metric,
							random_state=random_state, verbose=verbose)
		for e in estimators:
			check_estimator(e)
		self.n = len(estimators)
		self.base_estimators = estimators
		self.set_warm_start(warm_start)
		self.set_metric(metric)
		self.set_verbose(verbose)
		self.set_random_state(random_state)

	def _initialize(self):
		if self._is_fitted():
			for e in self.estimators_ : e.set_warm_start(self.warm_start)
		else:
			self.estimators_ = deepcopy(self.base_estimators)

	def fit(self, X, Y):
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
		TimeSeriesClassifier.set_verbose(self, verbose)
		estimators = self.estimators_ if hasattr(self, 'estimators_') else self.base_estimators
		for e in estimators:
			e.set_verbose(verbose-1)

	def set_random_state(self, random_state):
		TimeSeriesClassifier.set_random_state(self, random_state)
		estimators = self.estimators_ if hasattr(self, 'estimators_') else self.base_estimators
		for e in estimators:
			e.set_random_state(self.random_state.randint(0, 2**16))

	def set_metric(self, metric):
		TimeSeriesClassifier.set_metric(self, metric)
		estimators = self.estimators_ if hasattr(self, 'estimators_') else self.base_estimators
		for e in estimators:
			e.set_metric(metric)

	def set_warm_start(self, warm_start):
		TimeSeriesClassifier.set_warm_start(self, warm_start)
		estimators = self.estimators_ if hasattr(self, 'estimators_') else self.base_estimators
		for e in estimators:
			e.set_warm_start(warm_start)

	def _is_fitted(self):
		attributes = ["estimators_","n_classes_","n_features_"]
		return TimeSeriesClassifier._is_fitted(self, attributes=attributes)
