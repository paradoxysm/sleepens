"""Cross-Validation"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from tqdm import trange
from sklearn.utils import shuffle as sk_shuffle
from copy import deepcopy

from sleepens.analysis import get_metrics
from sleepens.utils import create_random_state, check_XY
from sleepens.ml import check_estimator, check_timeseries_estimator

def cross_validate(estimator, X, Y, cv=5, repeat=1, metric='accuracy', random_state=None,
					shuffle=True, verbose=0):
	"""
	Conduct a k-fold cross validation.

	Parameters
	----------
	estimator : Classifier
		Estimator to use.

	X : array-like
		Data to use for cross-validation.
		If using a Classifier, shape should be (n_samples, n_features).
		If using a TimeSeriesClassifier, shape should be (n_series, n_samples, n_features).

	Y : array-like
		Target labels to use for cross-validation.
		If using a Classifier, shape should be (n_samples,).
		If using a TimeSeriesClassifier, shape should be (n_series, n_samples,).

	cv : int, default=5
		Number of folds to use in k-fold cross-validation

	repeat : int, default=1
		Number of times to repeat cross-validation to average
		the results.

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
		If `random_state` is a RandomState, return that RandomState.

	shuffle : bool, default=True
		Determine if cross-validation should shuffle the data.

	verbose : int, default=0
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.

	Returns
	-------
	score : float
		The average score across the cross-validation.

	Y_hat : ndarray or list of ndarray
		The cross-validated predictions.
		If using a Classifier, `Y_hat` is an ndarray with a shape of (n_samples, n_classes)
		If using a TimeSeriesClassifier, `Y_hat` is a list of ndarrays with
		a shape of (n_series, n_samples, n_classes)
	"""
	if metric is None : metric = 'accuracy'
	metric = get_metrics(metric)
	random_state = create_random_state(random_state)
	check_estimator(estimator)
	if len(X) < cv:
		if len(X) == 1 : raise ValueError("X is length 1 and cross validation requires at least 2 samples/files")
		cv = len(X)
		if verbose > 1 : print("Found too small X, reducing cv to", cv)
	if verbose > 0 : print("Cross-validation with", cv, "folds", "and", repeat, "repeats")
	try:
		check_timeseries_estimator(estimator)
	except : return _cross_validate_classic(estimator, X, Y, cv, repeat, metric,
				random_state, shuffle, verbose)
	else : return _cross_validate_timeseries(estimator, X, Y, cv, repeat, metric,
				random_state, shuffle, verbose)

def _cross_validate_classic(estimator, X, Y, cv, repeat, metric, random_state,
					shuffle, verbose):
	"""
	Conduct a k-fold cross validation on a Classifier.

	Parameters
	----------
	estimator : Classifier
		Estimator to use.

	X : array-like, shape=(n_samples, n_features)
		Data to use for cross-validation.

	Y : array-like, shape=(n_samples,)
		Target labels to use for cross-validation.

	cv : int
		Number of folds to use in k-fold cross-validation

	repeat : int
		Number of times to repeat cross-validation to average
		the results.

	metric : Metric
		Metric to use for scoring.

	random_state : RandomState
		RandomState for shuffling.

	shuffle : bool
		Determine if cross-validation should shuffle the data.

	verbose : int
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.

	Returns
	-------
	score : float
		The average score across the cross-validation.

	Y_hat : ndarray, shape=(n_samples, n_classes)
		The cross-validated predictions.
	"""
	e_verbose = estimator.verbose
	suppressed_verbose = verbose - 1 if verbose >= 1 else 0
	estimator.set_verbose(suppressed_verbose)
	X, Y = check_XY(X=X, Y=Y)
	if Y.ndim == 1 : Y = Y.reshape(-1,1)
	Y_hat = np.zeros((len(Y), len(np.unique(Y))))
	for r in range(repeat):
		if verbose > 0 : print("Fitting on repeat", r+1)
		x_y = np.concatenate((X, Y, np.arange(len(X)).reshape(-1,1)), axis=1)
		if shuffle : random_state.shuffle(x_y)
		X_s, Y_s, i_s = x_y[:,:X.shape[-1]], x_y[:,X.shape[-1]:-1], x_y[:,-1]
		X_folds, Y_folds, i_folds = (np.array_split(X_s, cv),
									np.array_split(Y_s, cv),
									np.array_split(i_s, cv))
		if verbose == 1 : folds = trange(cv)
		else : folds = range(cv)
		for f in folds:
			if verbose > 1 : print("Fitting on fold", f+1)
			x_ = np.concatenate(X_folds[:-1]) if len(X_folds) > 2 else X_folds[0]
			y_ = np.concatenate(Y_folds[:-1]) if len(Y_folds) > 2 else Y_folds[0]
			e = deepcopy(estimator)
			e.fit(x_, y_)
			if verbose > 1 : print("Predicting for fold", f+1)
			Y_hat_ = e.predict_proba(X_folds[-1])
			if verbose > 2:
				score_ = metric.score(np.argmax(Y_hat_, axis=1), Y_folds[-1])
				print("Score:", score_)
			Y_hat[i_folds[-1].astype(int)] += Y_hat_ / repeat
			X_folds = X_folds[-1:] + X_folds[:-1]
			Y_folds = Y_folds[-1:] + Y_folds[:-1]
			i_folds = i_folds[-1:] + i_folds[:-1]
	estimator.set_verbose(e_verbose)
	p = np.argmax(Y_hat, axis=1)
	return metric.score(p, Y), Y_hat

def _cross_validate_timeseries(estimator, X, Y, cv, repeat, metric, random_state,
					shuffle, verbose):
	"""
	Conduct a k-fold cross validation on a TimeSeriesClassifier.

	Parameters
	----------
	estimator : TimeSeriesClassifier
		Estimator to use.

	X : list of array-like, shape=(n_series, n_samples, n_features)
		Data to use for cross-validation. Series can be of different lengths.

	Y : list of array-like, shape=(n_series, n_samples,)
		Target labels to use for cross-validation.

	cv : int
		Number of folds to use in k-fold cross-validation

	repeat : int
		Number of times to repeat cross-validation to average
		the results.

	metric : Metric
		Metric to use for scoring.

	random_state : RandomState
		RandomState for shuffling.

	shuffle : bool
		Determine if cross-validation should shuffle the data.

	verbose : int
		Determines the verbosity of cross-validation.
		Higher verbose levels result in more output logged.

	Returns
	-------
	score : float
		The average score across the cross-validation.

	Y_hat : list of ndarray, shape=(n_series, n_samples, n_classes)
		The cross-validated predictions.
	"""
	e_verbose = estimator.verbose
	suppressed_verbose = verbose - 1 if verbose >= 1 else 0
	estimator.set_verbose(suppressed_verbose)
	for x, y in zip(X, Y) : check_XY(X=x, Y=y)
	try : X_, Y_ = np.concatenate(X), np.concatenate(Y)
	except : raise ValueError("Inputs have different number of features")
	for i in range(len(Y)):
		if Y[i].ndim == 1 : Y[i] = Y[i].reshape(-1,1)
	Y_hat = [np.zeros((len(y), len(np.unique(y)))) for y in Y]
	lens = [len(y) for y in Y]
	for r in range(repeat):
		if verbose > 0 : print("Fitting on repeat", r+1)
		X_s, Y_s, i_s = sk_shuffle(X, Y, np.arange(len(X), dtype=int), random_state=random_state)
		X_folds, Y_folds, i_folds = (np.array_split(np.array(X_s, dtype=object), cv),
									np.array_split(np.array(Y_s, dtype=object), cv),
									np.array_split(np.array(i_s, dtype=object), cv))
		if verbose == 1 : folds = trange(cv)
		else : folds = range(cv)
		for f in folds:
			if verbose > 1 : print("Fitting on fold", f+1)
			x_ = np.concatenate(X_folds[:-1]) if len(X_folds) > 2 else X_folds[0]
			y_ = np.concatenate(Y_folds[:-1]) if len(Y_folds) > 2 else Y_folds[0]
			e = deepcopy(estimator)
			e.fit(x_, y_)
			if verbose > 1 : print("Predicting for fold", f+1)
			Y_hat_ = e.predict_proba(X_folds[-1])
			if verbose > 2:
				scores_, lens_ = [], [len(y_f) for y_f in Y_folds[-1]]
				for y_hat, y in zip(Y_hat_, Y_folds[-1]):
					p = np.argmax(y_hat, axis=1)
					scores_.append(metric.score(p, y))
				score_ = np.average(scores_, weights=lens_)
				print("Score:", score_)
			for i in range(len(Y_hat_)):
				Y_hat[i_folds[-1][i]] += Y_hat_[i] / repeat
			X_folds = X_folds[-1:] + X_folds[:-1]
			Y_folds = Y_folds[-1:] + Y_folds[:-1]
			i_folds = i_folds[-1:] + i_folds[:-1]
	estimator.set_verbose(e_verbose)
	scores = []
	for y_hat, y in zip(Y_hat, Y):
		p = np.argmax(y_hat, axis=1)
		scores.append(metric.score(p, y))
	score = np.average(scores, weights=lens)
	return score, Y_hat
