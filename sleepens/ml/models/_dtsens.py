import numpy as np
from tqdm import trange
from copy import deepcopy
from sklearn.neighbors import KDTree

from sleepens.ml import cross_validate, check_estimator
from sleepens.ml._base_model import TimeSeriesClassifier
from sleepens.ml.models import GradientBoostingClassifier
from sleepens.utils import time_array

default_estimators = [GradientBoostingClassifier() for i in range(10)]

# Deprecated
class DynamicTimeSeriesEnsemble(TimeSeriesClassifier):
	def __init__(self, estimators=default_estimators,
					k=100, select=3, threshold=0.05, rank=False, cv=5,
					warm_start=False, metric='accuracy', random_state=None, verbose=0):
		TimeSeriesClassifier.__init__(self, warm_start=warm_start, metric=metric,
							random_state=random_state, verbose=verbose)
		for e in estimators:
			check_estimator(e)
		if not isinstance(self.cv, int):
			raise ValueError("CV parameter must be an integer")
		self.n = len(estimators)
		self.base_estimators = estimators
		self.k = k
		self.select = select
		self.threshold = threshold
		self.rank = rank
		self.cv = cv

	def _initialize(self):
		if self._is_fitted():
			for e in self.estimators_ : e.warm_start = self.warm_start
		else:
			self.estimators_ = deepcopy(self.base_estimators)

	def fit(self, X, Y):
		X_, Y_ = time_sequence(X, self.n), time_sequence(Y, self.n)
		X_, Y_ = self._fit_setup(X_, Y_)
		X, Y = time_array(X_, self.n), time_array(Y_, self.n)
		if X.shape[1] % self.n != 0:
			raise ValueError("Input was not made for", self.n, "length TimeSeriesEnsemble")
		if len(Y.shape) != 2 or Y.shape[1] != self.n:
			raise ValueError("Input was not made for", self.n, "length TimeSeriesEnsemble")

		X, Y = self._fit_setup(X, Y)
		X_a, Y_a = [time_array(x, self.n) for x in X], [time_array(y, self.n) for y in Y]
		X_a, Y_a = np.concatenate(X_a), np.concatenate(Y_a)

		self.tree_ = KDTree(X_)
		self.tree_scores_ = np.zeros((len(X_), self.n))
		if self.verbose == 1 or self.verbose == 2 : stages = trange(self.n, desc="Estimators")
		else : stages = range(self.n)
		for s in stages:
			if self.verbose > 2 : print("Fitting estimator", s)
			self.estimators_[s].fit(X, Y[:,s])
		if self.verbose == 1 or  self.verbose == 2 : stages = trange(self.n, desc="DES")
		else : stages = range(self.n)
		for s in stages:
			if self.verbose > 2 : print("Cross-validating estimator", s)
			y = Y[:,s]
			_, y_hat = cross_validate(e, X, y, cv=self.cv, metric=self.metric,
						random_state=self.random_state, verbose=self.verbose-2)
			p = np.argmax(y_hat, axis=1)
			score = np.where(p == y, 1, 0)
			self.tree_scores_[s:len(X)+s, s] = score
		if self.verbose > 1 : print("Completed training")
		return self

	def predict_proba(self, X):
		X_ = time_sequence(X, self.n)
		X_ = self._predict_setup(X_)
		X = time_array(X_, self.n)
		if X.shape[1] % self.n != 0:
			raise ValueError("Input was not made for", self.n, "length TimeSeriesEnsemble")
		neighbors = self.tree_.query(X_, k=self.k, return_distance=False)
		accuracies = np.mean(self.tree_scores_[neighbors], axis=1)
		if self.select is None : self.select = len(self.estimators_)
		sort = np.argsort(accuracies, axis=-1)
		selected = sort[:,-self.select:]
		s_accuracies = np.take_along_axis(accuracies, selected, axis=-1)
		i = 0
		while i < self.n:
			if i < self.select:
				max = np.max(s_accuracies, axis=1)
				min = np.argmin(s_accuracies, axis=1)
				avg = np.mean(s_accuracies, axis=1)
				poor = np.where(max - avg >= self.threshold)[0]
				selected[poor, min[poor]] = sort[poor, -i-1]
				s_accuracies = np.take_along_axis(accuracies, selected, axis=-1)
			indices = np.repeat(np.arange(len(selected)),self.select).reshape(-1,self.select)
			low = np.where(selected > indices)
			selected[low] = sort[low[0], -i-1]
			high = np.where(indices > len(selected) - self.n + selected)
			selected[high] = sort[high[0], -i-1]
			i += 1
		Y_hat = np.zeros((len(X_), self.n_classes_))
		count = np.zeros((len(X_),), dtype=int)
		if self.verbose == 1 or self.verbose == 2 : stages = trange(self.n)
		else : stages = range(self.n)
		for s in stages:
			if self.verbose > 2 : print("Predicting on estimator", s)
			i = np.where(selected == s)
			rows = i[0]
			if len(rows) == 0 : continue
			p = self.estimators_[s].predict_proba(X[rows-s])
			c = np.ones(len(rows))
			if self.rank:
				p *= s_accuracies[i]
				c *= s_accuracies[i]
			np.add.at(Y_hat, rows, p)
			np.add.at(count, rows, c)
		Y_hat /= count.reshape(-1,1)
		if self.verbose > 1 : print("Completed predicting")
		return Y_hat

	def _is_fitted(self):
		attributes = ["estimators_","n_classes_","n_features_",
						"tree_","tree_scores_"]
		return Classifier._is_fitted(self, attributes=attributes)
