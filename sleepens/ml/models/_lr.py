from sklearn.linear_model import LogisticRegression as lrm

from sleepens.ml._base_model import Classifier

class LogisticRegression(Classifier):
	def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
				intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
				max_iter=100, multi_class='auto', verbose=0, warm_start=False,
				n_jobs=None, l1_ratio=None, metric='accuracy'):
		Classifier.__init__(self, warm_start=warm_start, metric=metric,
							random_state=random_state, verbose=verbose)
		self.lr = lrm(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
					intercept_scaling=intercept_scaling, class_weight=class_weight,
					random_state=random_state, solver=solver, max_iter=max_iter,
					multi_class=multi_class, verbose=verbose, warm_start=warm_start,
					n_jobs=n_jobs, l1_ratio=l1_ratio)

	def fit(self, X, Y):
		X, Y = self._fit_setup(X, Y)
		return self.lr.fit(X, Y)

	def predict_proba(self, X):
		X = self._predict_setup(X)
		return self.lr.predict_proba(X)

	def _is_fitted(self):
		attributes = ["coef_","intercept_"]
		return Classifier._is_fitted(self.lr, attributes=attributes) and \
				Classifier._is_fitted(self)
