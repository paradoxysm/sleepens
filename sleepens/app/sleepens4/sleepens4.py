import numpy as np
from copy import deepcopy

from sleepens.app._base_sleepens import AbstractSleepEnsemble

from sleepens.io.reader import smrMATReader
from sleepens.app.sleepens4.params import params
from sleepens.app.sleepens4.processor import Sleepens4_Processor
from sleepens.analysis import get_metrics
from sleepens.ml import cross_validate
from sleepens.ml.models import StackedTimeSeriesEnsemble, GradientBoostingClassifier, TimeSeriesEnsemble
from sleepens.addon import RemFix


class SleepEnsemble4(AbstractSleepEnsemble):
	def __init__(self, params=params, verbose=0):
		AbstractSleepEnsemble.__init__(self, params=params, verbose=verbose)
		self.name = "Sleepens4"
		self.reader = smrMATReader(verbose=verbose)
		self.processor = Sleepens4_Processor(params=params['process'], verbose=verbose)
		gbc = GradientBoostingClassifier(n_estimators=250, max_features='sqrt', learning_rate=0.05,
										subsample=0.75, max_depth=5)
		tsens1 = TimeSeriesEnsemble(estimators=[deepcopy(gbc) for i in range(10)])
		tsens2 = TimeSeriesEnsemble(estimators=[deepcopy(gbc) for i in range(10)])
		self.classifier = StackedTimeSeriesEnsemble(layer_1=tsens1, layer_2=tsens2, warm_start=False,
								metric='accuracy', random_state=None, verbose=verbose)

	def read(self, file, labels=False):
		eeg = self.reader.read_data(file, self.params['reader']['EEG_NAME'])
		neck = self.reader.read_data(file, self.params['reader']['NECK_NAME'])
		mass = self.reader.read_data(file, self.params['reader']['MASS_NAME'])
		if labels:
			labels = self.reader.read_labels(file, self.params['reader']['SCORE_NAME'],
									map=self.params['reader']['SCORE_MAP'])
		else : labels = None
		data = (eeg, neck, mass)
		return data, labels

	def process(self, data, labels, name):
		ds = self.processor.process(data, labels, name)
		return ds

	def fit(self, X, Y):
		self.classifier.fit(X, Y)
		return self

	def predict(self, X):
		if not self._is_fitted():
			raise RuntimeError("Sleep Ensemble has not been fitted")
		Y_hat = self.classifier.predict_proba(X)
		if isinstance(Y_hat, list):
			p = [model._addon(y) for y in Y_hat]
		else : p = model._addon(Y_hat)
		return p, Y_hat

	def cross_validate(self, X, Y, cv=5, repeat=1, metric='accuracy', random_state=None,
						shuffle=True):
		if len(X) < cv:
			if len(X) == 1 : raise ValueError("X is length 1 and cross validation requires at least 2 samples/files")
			cv = len(X)
			if self.verbose > 1 : print("Found too small X, reducing cv to", cv)
		_, Y_hat = cross_validate(self.classifier, X, Y, cv=cv, repeat=repeat, metric=metric,
						random_state=random_state, shuffle=shuffle, verbose=self.verbose)
		if isinstance(Y_hat, list):
			p = [self._addon(y) for y in Y_hat]
		else:
			p = [self._addon(Y_hat)]
		return p, Y_hat

	def score(self, p, Y, metric='accuracy'):
		metric = get_metrics(metric)
		if isinstance(p, list) and isinstance(Y, list):
			score = [metric.score(p_, y_) for p_, y_ in zip(p, Y)]
		elif not isinstance(p, list) and isinstance(Y, list):
			score = [metric.score(p, Y)]
		else : raise ValueError("Both p and Y must either be lists of ndarrays or ndarrays")
		return score

	def set_verbose(self, verbose):
		self.verbose = verbose
		self.reader.verbose = verbose
		self.processor.verbose = verbose
		self.classifier.set_verbose(verbose)

	def _addon(self, Y_hat):
		p = np.argmax(Y_hat, axis=1)
		return RemFix(verbose=self.verbose).addon(Y_hat, p)
