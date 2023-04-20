from sklearn.ensemble import GradientBoostingClassifier as gbm

from sleepens.ml._base_model import Classifier

class GradientBoostingClassifier(Classifier):
	"""
	Wrapper class for sci-kit learn's Gradient Boosting Classifier.

    Gradient Boosting Classifier builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage `n_classes_`
    regression trees are fit on the negative gradient of the
    binomial or multinomial deviance loss function. Binary classification
    is a special case where only a single regression tree is induced.

    Parameters
    ----------
    loss : {'deviance', 'exponential'}, default='deviance'
        The loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm.

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    criterion : {'friedman_mse', 'mse'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria
        are 'friedman_mse' for the mean squared error with improvement
        score by Friedman, 'mse' for mean squared error, and 'mae' for
        the mean absolute error. The default value of 'friedman_mse' is
        generally the best as it can provide a better approximation in
        some cases.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_depth : int, default=3
        The maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where `N` is the total number of samples, `N_t` is the number of
        samples at the current node, `N_t_L` is the number of samples in the
        left child, and `N_t_R` is the number of samples in the right child.
        `N`, `N_t`, `N_t_R` and `N_t_L` all refer to the weighted sum,
        if `sample_weight` is passed.

    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        `init` has to provide :meth:`fit` and :meth:`predict_proba`. If
        'zero', the initial raw predictions are set to zero. By default, a
        `DummyEstimator` predicting the classes priors is used.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split. It also controls the random spliting of the training data
		to obtain a validation set if `n_iter_no_change` is not None.
        Pass an int for reproducible output across multiple function calls.

    max_features : {'auto', 'sqrt', 'log2'}, int or float, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If 'auto', then `max_features=sqrt(n_features)`.
        - If 'sqrt', then `max_features=sqrt(n_features)`.
        - If 'log2', then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than `max_features` features.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    max_leaf_nodes : int, default=None
        Grow trees with `max_leaf_nodes` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    warm_start : bool, default=False
        When set to `True`, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if `n_iter_no_change` is set to an integer.

    n_iter_no_change : int, default=None
        `n_iter_no_change` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside `validation_fraction` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous `n_iter_no_change` numbers of
        iterations. The split is stratified.

    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for `n_iter_no_change` iterations (if set to a
        number), the training stops.

    ccp_alpha : non-negative float, default=0.0
    	Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        `ccp_alpha` will be chosen. By default, no pruning is performed.

    Attributes
    ----------
	gb : sklearn.ensemble.GradientBoostingClassifier
		The underlying GradientBoostingClassifier.

	n_classes_ : int
		Number of classes.

	n_features_ : int
		Number of features.

	Underlying Attributes
	---------------------
	n_estimators_ : int
    	The number of estimators as selected by early stopping (if
        `n_iter_no_change` is specified). Otherwise it is set to
        `n_estimators`.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values).

    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        `oob_improvement_[0]` is the improvement in
        loss of the first stage over the `init` estimator.
        Only available if `subsample < 1.0`.

    train_score_ : ndarray of shape (n_estimators,)
        The i-th score `train_score_[i]` is the deviance (= loss) of the
        model at iteration `i` on the in-bag sample.
        If `subsample == 1` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete `LossFunction` object.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the `init` argument or `loss.init_estimator`.

    estimators_ : ndarray of DecisionTreeRegressor, shape=(n_estimators, `loss_.K`)
        The collection of fitted sub-estimators. `loss_.K` is 1 for binary
        classification, otherwise n_classes.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_ : int
        The number of data features.

    n_classes_ : int
        The number of classes.

    max_features_ : int
        The inferred value of max_features.
	"""
	def __init__(self, loss='log_loss', learning_rate=0.1, n_estimators=100,
				subsample=1.0, criterion='friedman_mse', min_samples_split=2,
				min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
				min_impurity_decrease=0.0, init=None,
				random_state=None, max_features=None, verbose=0, max_leaf_nodes=None,
				warm_start=False, validation_fraction=0.1, n_iter_no_change=None,
				tol=0.0001, ccp_alpha=0.0):
		Classifier.__init__(self, warm_start=warm_start, random_state=random_state, verbose=verbose)
		self.gb = gbm(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
					subsample=subsample, criterion=criterion, min_samples_split=min_samples_split,
					min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
					max_depth=max_depth, min_impurity_decrease=min_impurity_decrease,
					init=init, random_state=random_state,
					max_features=max_features, verbose=verbose, max_leaf_nodes=max_leaf_nodes,
					warm_start=warm_start, validation_fraction=validation_fraction,
					n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

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
		self : Classifier
			Fitted classifier.
		"""
		X, Y = self._fit_setup(X, Y)
		return self.gb.fit(X, Y)

	def predict_proba(self, X):
		"""
		Return the prediction probabilities on the given data.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Probability predictions.
		"""
		X = self._predict_setup(X)
		return self.gb.predict_proba(X)

	def _is_fitted(self):
		"""
		Returns if the Classifier has been trained and is
		ready to predict new data.

		Returns
		-------
		fitted : bool
			True if the Classifier is fitted, False otherwise.
		"""
		attributes = ["estimators_","n_classes_","n_features_in_"]
		return Classifier._is_fitted(self.gb, attributes=attributes)
