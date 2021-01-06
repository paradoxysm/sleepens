"""Randomized Search with Cross-Validation"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from copy import deepcopy
from sklearn.model_selection import ParameterSampler

from sleepens.analysis import get_metrics
from sleepens.utils import create_random_state, check_XY
from sleepens.ml import cross_validate, check_estimator

class RandomizedSearch:
	"""
	Conduct a Randomized Search for hyperparameter optimization.

	Parameters
	----------
	estimator : Classifier
		Estimator to use.

	distributions : dict
		Dictionary of distributions to sample hyperparameters.
		Keys use attribute names, while items are either
		array-likes or distributions. When using array-likes,
		RandomizedSearch will sample assuming uniform distribution.

	n_iter : int, default=50
		Number of hyperparameter combinations to run.

	cv : int, default=5
		Number of folds to use in k-fold cross-validation

	repeat : int, default=1
		Number of times to repeat cross-validation to average
		the results.

	shuffle : bool, default=True
		Determine if cross-validation should shuffle the data.

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
		Determines the verbosity of the RandomizedSearch.
		Higher verbose levels result in more output logged.

	Attributes
	----------
	params_ : list of dict
		List of all hyperparameter combinations used in order
		on the last call to fit.

	scores_ : list
		List of scores from each hyperparameter combination
		in the same order as `params_`.

	best_params_ : dict
		The best hyperparameter combination found in the last call to fit.

	best_score_ : float
		The corresponding score of the best hyperparameter combination
		found in the last call to fit.
	"""
	def __init__(self, estimator, distributions, n_iter=50, cv=5, repeat=1,
					shuffle=True, metric='accuracy', random_state=None, verbose=0):
		check_estimator(estimator)
		self.estimator = estimator
		self.distributions = distributions
		self.n_iter = n_iter
		self.cv = cv
		self.repeat = repeat
		self.shuffle = shuffle
		if metric is None : metric = 'accuracy'
		self.metric = get_metrics(metric)
		self.random_state = create_random_state(random_state)
		self.verbose = verbose

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
		self : RandomizedSearch
			RandomizedSearch object that contains the
			hyperparameter combinations explored and corresponding scores,
			along with the best combination and corresponding score.
		"""
		self.params_ = list(ParameterSampler(self.distributions, self.n_iter, random_state=self.random_state))
		if self.verbose > 0 : print("Randomized Search on", len(self.params_), "candidates using",
								str(self.cv)+"-fold cross-validation with", self.repeat, "repeats")
		self.scores_ = []
		self.best_params_, self.best_score_ = None, -1
		if self.verbose == 1 : iter = trange(len(self.params_))
		else : iter = range(len(self.params_))
		for i in iter:
			params = self.params_[i]
			e = deepcopy(self.estimator)
			for k, v in params.items() : setattr(e, k, v)
			if self.verbose > 1 : print("Evaluating", params)
			score, _ = cross_validate(e, X, Y, cv=self.cv, repeat=self.repeat,
							metric=self.metric, random_state=self.random_state,
							shuffle=self.shuffle, verbose=self.verbose-1)
			if self.verbose > 1 : print("Score:", score)
			self.scores_.append(score)
			if score > self.best_score_ : self.best_params_, self.best_score_ = params, score
		if self.verbose > 0 : print("Completed Randomized Search")
		return self
