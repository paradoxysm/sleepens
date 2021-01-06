from sklearn.ensemble import RandomForestClassifier as rfm

from sleepens.ml._base_model import Classifier

class RandomForestClassifier(Classifier):
	def __init__(self, n_estimators=100, criterion='gini', max_depth=None,
				min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
				max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
				min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
				random_state=None, verbose=0, warm_start=False, class_weight=None,
				ccp_alpha=0.0, max_samples=None, metric='accuracy'):
		Classifier.__init__(self, warm_start=warm_start, metric=metric,
							random_state=random_state, verbose=verbose)
		self.rf = rfm(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
					min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
					min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
					max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
					min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score,
					n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
					class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)

	def fit(self, X, Y):
		X, Y = self._fit_setup(X, Y)
		return self.rf.fit(X, Y)

	def predict_proba(self, X):
		X = self._predict_setup(X)
		return self.rf.predict_proba(X)

	def _is_fitted(self):
		attributes = ["estimators_","n_classes_","n_features_"]
		return Classifier._is_fitted(self.rf, attributes=attributes)
