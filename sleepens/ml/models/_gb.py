from sklearn.ensemble import GradientBoostingClassifier as gbm

from sleepens.ml._base_model import Classifier

class GradientBoostingClassifier(Classifier):
	def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
				subsample=1.0, criterion='friedman_mse', min_samples_split=2,
				min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
				min_impurity_decrease=0.0, min_impurity_split=None, init=None,
				random_state=None, max_features=None, verbose=0, max_leaf_nodes=None,
				warm_start=False, validation_fraction=0.1, n_iter_no_change=None,
				tol=0.0001, ccp_alpha=0.0, metric='accuracy'):
		Classifier.__init__(self, warm_start=warm_start, metric=metric,
							random_state=random_state, verbose=verbose)
		self.gb = gbm(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
					subsample=subsample, criterion=criterion, min_samples_split=min_samples_split,
					min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
					max_depth=max_depth, min_impurity_decrease=min_impurity_decrease,
					min_impurity_split=min_impurity_split, init=init, random_state=random_state,
					max_features=max_features, verbose=verbose, max_leaf_nodes=max_leaf_nodes,
					warm_start=warm_start, validation_fraction=validation_fraction,
					n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

	def fit(self, X, Y):
		X, Y = self._fit_setup(X, Y)
		return self.gb.fit(X, Y)

	def predict_proba(self, X):
		X = self._predict_setup(X)
		return self.gb.predict_proba(X)

	def _is_fitted(self):
		attributes = ["estimators_","n_classes_","n_features_"]
		return Classifier._is_fitted(self.gb, attributes=attributes)
