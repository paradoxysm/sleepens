
from abc import ABC, abstractmethod
from sleepens.utils._base import Base

class AbstractSleepEnsemble(Base, ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, Y, weight=None):
        raise NotImplementedError("No fit function implemented")

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("No predict function implemented")

    @abstractmethod
    def predict_log_proba(self, X):
        raise NotImplementedError("No predict_log_proba function implemented")

    @abstractmethod
    def predict_proba(self, X):
        raise NotImplementedError("No predict_proba function implemented")

    @abstractmethod
    def process(self, filepath, labels=False):
        raise NotImplementedError("No process function implemented")

    @abstractmethod
    def export_model(self):
        raise NotImplementedError("No export_model function implemented")

    @abstractmethod
    def load_model(self, filepath):
        raise NotImplementedError("No load_model function implemented")
