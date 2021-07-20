from abc import ABC, abstractmethod
import joblib

from sleepens.io._base_reader import AbstractReader, check_reader
from sleepens.process._base_processor import AbstractProcessor, check_processor
from sleepens.ml import check_estimator
from sleepens.ml._base_model import Classifier


def check_sleepens(object):
	if not issubclass(type(object), AbstractSleepEnsemble):
		raise ValueError("Object is not a Sleep Ensemble")

class AbstractSleepEnsemble(ABC):
	def __init__(self, params=None, verbose=0):
		self.name = "AbstractSleepEnsemble"
		self.params = params
		self.verbose = verbose
		self.reader = None
		self.processor = None
		self.classifier = None

	@abstractmethod
	def read(self, file, labels=False):
		raise NotImplementedError("No read function implemented")

	@abstractmethod
	def process(self, file, labels=False):
		raise NotImplementedError("No process function implemented")

	@abstractmethod
	def fit(self, X, Y):
		raise NotImplementedError("No fit function implemented")

	@abstractmethod
	def predict(self, X):
		raise NotImplementedError("No predict function implemented")

	@abstractmethod
	def cross_validate(self, X, Y, cv=5, repeat=1, metric='accuracy', random_state=None,
						shuffle=True):
		raise NotImplementedError("No cross_validate function implemented")

	def load(self, filepath):
		if self.verbose > 0 : print("Loading", filepath)
		sleepens = joblib.load(filepath)
		if issubclass(type(sleepens), AbstractSleepEnsemble) and sleepens.check() : return sleepens
		raise ValueError("Object at " + filepath + " not valid")

	def export(self, filepath):
		if self.verbose > 0 : print("Exporting to", filepath)
		joblib.dump(self, filepath)

	def check(self):
		check_reader(self.reader)
		check_processor(self.processor)
		check_estimator(self.classifier)
		return True

	def set_verbose(self, verbose):
		self.verbose = verbose

	def set_random_state(self, random_state):
		TimeSeriesClassifier.set_random_state(self, random_state)

	def set_metric(self, metric):
		TimeSeriesClassifier.set_metric(self, metric)

	def set_warm_start(self, warm_start):
		TimeSeriesClassifier.set_warm_start(self, warm_start)

	def _is_fitted(self, attributes=["n_classes_","n_features_"]):
		return self.classifier._is_fitted(self, attributes=attributes)


class ShellSleepEnsemble(AbstractSleepEnsemble):
	def __init__(self, params=None, verbose=0):
		AbstractSleepEnsemble.__init__(self, params=params, verbose=verbose)
		self.name = "ShellSleepEnsemble"

	def read(self, file, labels=False):
		raise NotImplementedError("This is a shell class")

	def process(self, file, labels=False):
		raise NotImplementedError("This is a shell class")

	def fit(self, X, Y):
		raise NotImplementedError("This is a shell class")

	def predict(self, X):
		raise NotImplementedError("This is a shell class")

	def cross_validate(self, X, Y, cv=5, repeat=1, metric='accuracy', random_state=None,
						shuffle=True):
		raise NotImplementedError("This is a shell class")
