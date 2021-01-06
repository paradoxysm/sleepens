import numpy as np
from tqdm import trange
from sklearn.neighbors import KDTree

from sleepens.ml import check_estimator
from sleepens.ml._base_model import Classifier
from sleepens.utils import check_XY


class DynamicEnsembleKNN(Classifier):
	def __init__(self, estimators, k=100, select=3, threshold=0.05, rank=False, cv=5,
					warm_start=False, metric='accuracy', random_state=None, verbose=0):
		Classifier.__init__(self, warm_start=warm_start, metric=metric,
							random_state=random_state, verbose=verbose)
		for e in estimators:
			check_estimator(e)
		if not isinstance(self.cv, int):
			raise ValueError("CV parameter must be an integer")
		self.estimators = estimators
		self.k = k
		self.select = select
		self.threshold = threshold
		self.rank = rank
		self.cv = cv

	def _initialize(self):
		if self._is_fitted():
			for e in self.estimators_ : e.warm_start = self.warm_start

	def fit(self, X, Y):
		X, Y = self._fit_setup(X, Y)
		self.tree_ = KDTree(X)
		self.tree_scores_ = np.zeros((len(X), len(self.estimators)))
		if self.verbose == 1 or self.verbose == 2:
			estimators = trange(len(self.estimators), desc="Estimators")
		else : estimators = range(len(self.estimators))
		for e in estimators:
			if self.verbose > 2 : print("Fitting estimator", e)
			self.estimators[e].fit(X, Y)
		if self.verbose == 1 or  self.verbose == 2 : estimators = trange(len(self.estimators), desc="DES")
		else : estimators = range(len(self.estimators))
		for e in estimators:
			if self.verbose > 2 : print("Cross-validating estimator", e)
			_, p = cross_validate(e, X, Y, cv=self.cv, metric=self.metric,
						random_state=self.random_state, verbose=self.verbose-2)
			score = np.where(p == Y, 1, 0)
			self.tree_scores_[:, e] = score
		if self.verbose > 1 : print("Completed training")
		return self

	def predict_proba(self, X):
		X = self._predict_setup(X)
		neighbors = self.tree_.query(X, k=self.k, return_distance=False)
		accuracies = np.mean(self.tree_scores_[neighbors], axis=1)
		if self.select is None : self.select = len(self.estimators_)
		sort = np.argsort(accuracies, axis=-1)
		selected = sort[:,-self.select:]
		s_accuracies = np.take_along_axis(accuracies, selected, axis=-1)
		for i in range(self.select):
			max = np.max(s_accuracies, axis=1)
			min = np.argmin(s_accuracies, axis=1)
			avg = np.mean(s_accuracies, axis=1)
			poor = np.where(max - avg >= self.threshold)[0]
			if len(poor) == 0 : break
			selected[poor, min[poor]] = sort[poor, -i-1]
			s_accuracies = np.take_along_axis(accuracies, selected, axis=-1)
		Y_hat = np.zeros((len(X), self.n_classes_))
		count = np.zeros((len(X),), dtype=int)
		if self.verbose == 1 or self.verbose == 2:
			estimators = trange(len(self.estimators))
		else : estimators = range(len(self.estimators))
		for e in estimators:
			if self.verbose > 2 : print("Predicting on estimator", e)
			i = np.where(selected == e)
			rows = i[0]
			if len(rows) == 0 : continue
			p = self.estimators_[e].predict_proba(X_[rows])
			c = np.ones(len(rows))
			if self.rank:
				p *= s_accuracies[i]
				c *= s_accuracies[i]
			np.add.at(Y_hat, rows, p)
			np.add.at(count, rows, c)
		Y_hat /= count
		if self.verbose > 1 : print("Completed predicting")
		return Y_hat

	def _is_fitted(self):
		if len(self.estimators) == 0 : return False
		attributes = ["n_classes_","n_features_",
						"tree_","tree_scores_"]
		return Classifier._is_fitted(self, attributes=attributes)
