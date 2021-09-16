import numpy as np
from copy import deepcopy
from sklearn import metrics

from sleepens.protocols.sleepens4.params import params
from sleepens.protocols.sleepens4.processor import process as processor
from sleepens.ml import cross_validate
from sleepens.ml.models import StackedTimeSeriesEnsemble, GradientBoostingClassifier, TimeSeriesEnsemble
from sleepens import postprocess


class SleepEnsemble4:
	def __init__(self, params=params, verbose=0):
		self.name = "Sleepens4"
		self.params = params
		self.verbose = verbose
		gbc = GradientBoostingClassifier(**params['classifier']['gbc'])
		tsens1 = TimeSeriesEnsemble(estimators=[deepcopy(gbc) for i in range(10)])
		tsens2 = TimeSeriesEnsemble(estimators=[deepcopy(gbc) for i in range(10)])
		self.classifier = StackedTimeSeriesEnsemble(layer_1=tsens1, layer_2=tsens2, warm_start=False,
								random_state=None, verbose=verbose)

	def read(self, reader, file, labels=False):
		eeg = reader.read_data(file, self.params['reader']['EEG_NAME'])
		neck = reader.read_data(file, self.params['reader']['NECK_NAME'])
		mass = reader.read_data(file, self.params['reader']['MASS_NAME'])
		if labels:
			labels = reader.read_labels(file, self.params['reader']['SCORE_NAME'],
									map=self.params['reader']['SCORE_MAP'])
		else : labels = None
		data = (eeg, neck, mass)
		return data, labels

	def process(self, data, labels, name):
		ds = processor(data, labels, name, self.params['process'], verbose=self.verbose)
		return ds

	def fit(self, X, Y):
		self.classifier.fit(X, Y)
		return self

	def predict(self, X):
		if not self.classifier._is_fitted():
			raise RuntimeError("Sleep Ensemble has not been fitted")
		Y_hat = self.classifier.predict_proba(X)
		if isinstance(Y_hat, list):
			p = [self._postprocess(y) for y in Y_hat]
		else : p = self._postprocess(Y_hat)
		return p, Y_hat

	def cross_validate(self, X, Y, cv=5, repeat=1, random_state=None,
						shuffle=True):
		if len(X) < cv:
			if len(X) == 1 : raise ValueError("X is length 1 and cross validation requires at least 2 samples/files")
			cv = len(X)
			if self.verbose > 1 : print("Found too small X, reducing cv to", cv)
		_, Y_hat = cross_validate(self.classifier, X, Y, cv=cv, repeat=repeat,
						random_state=random_state, shuffle=shuffle, verbose=self.verbose)
		if isinstance(Y_hat, list):
			p = [self._postprocess(y) for y in Y_hat]
		else:
			p = [self._postprocess(Y_hat)]
		return p, Y_hat

	def score(self, p, Y):
		if isinstance(p, list) and isinstance(Y, list):
			score = [metrics.accuracy_score(y_, p_) for p_, y_ in zip(p, Y)]
		elif not isinstance(p, list) and isinstance(Y, list):
			score = [metrics.accuracy_score(Y, p)]
		else : raise ValueError("Both p and Y must either be lists of ndarrays or ndarrays")
		return score

	def set_verbose(self, verbose):
		self.verbose = verbose
		self.classifier.set_verbose(verbose)

	def _postprocess(self, Y_hat):
		p = np.argmax(Y_hat, axis=1)
		p = postprocess.WakeMax(Y_hat, p, map=self.params['process']['INVERSE_MAP'])
		p = postprocess.WakeToREM(Y_hat, p, map=self.params['process']['INVERSE_MAP'])
		p = postprocess.REMDrop(Y_hat, p, map=self.params['process']['INVERSE_MAP'], window=3, threshold=0.1)
		p = postprocess.MinREM(Y_hat, p, map=self.params['process']['INVERSE_MAP'], min_rem=3)
		p = postprocess.TransitionFix(Y_hat, p, transitions=self.params['transition'])
		return p
