import numpy as np

from sklearn.mixture import GaussianMixture as GMM
from sleep.utils.selection import GridSearchCV
from sleep.utils.misc import separate_by_label

def generative_oversample(data, labels, desired=None, balance='auto',
							seed=None, verbose=True):
	"""
	Generative Oversampling of data to improve imbalanced
	datasets.

	Model a generative algorithm over each of the
	desired labelled data to sample more data from
	the resulting distributions.

	Parameters
	----------
	data : array-like, shape=(n_samples, n_features)
		Data.

	labels : array-like, shape=(n_samples,)
		Labels corresponding to data.

	desired : {None, array-like}, default=None
		List of labels to oversample.
		If None, balances all labels.

	balance : {'auto', int, dict}, default='auto'
		Determines how to balance and oversample.
		Must be one of:
		 - 'auto' : Automatically balance the dataset
		 			by oversampling `desired` labels to
					match the number of samples in the majority
					label. Recommended to set `desired` to None,
					or risk not oversampling a minority label.
		 - int : Oversample `desired` labels so that total
		 			such samples reach this value.
		 - dict : Oversample each label up to the values in
		 			the dictionary.

	cluster : generative clustering algorithm, default=GMM
		Generative clustering algorithm. Must have a
		sampling function, `sample`, to sample new data.
		This sampling function must return a list-convertable
		list of samples.

	search_space : dict, default={}
		Dictionary of parameters and search spaces for
		a Grid Search of the clustering algorithm.

	Returns
	-------
	data : ndarray
		All oversampled data, shuffled.

	labels : ndarray
		All corresponding labels.
	"""
	if len(data) == 0:
		if verbose : print("No data to oversample")
		return data, labels
	if verbose : print("Generative Oversampling")
	data = separate_by_label(data, labels)
	separated = data
	if desired is not None:
		data_labels = list(data.keys())
		for d in desired:
			if desired not in  data_labels:
				raise ValueError("Desired label is not in the data")
		separated = {d: separated[d] for d in desired}
	if balance == 'auto' or isinstance(balance, int):
		if balance == 'auto':
			if verbose : print("Balancing Dataset. Method set to 'auto'")
			target = np.max([len(data[k]) for k in data.keys()])
		else:
			if verbose : print("Balancing Dataset up to", str(balance))
			target = balance
		n_samples = {k: target - len(separated[k]) for k in separated.keys()}
		n_samples = {k: 0 if n_samples[k] < 0 else n_samples[k] for k in separated.keys()}
	elif isinstance(balance, str) and balance[0] == 'f' and balance[1:].isdigit():
		if verbose : print("Upscaling Dataset by a factor of", balance[1:])
		factor = int(balance[1:])
		n_samples = {k: factor * len(separated[k]) for k in separated.keys()}
	elif isinstance(balance, (dict)):
		if verbose : print("Balancing Dataset. Method set to custom")
		n_samples = balance
	else:
		raise ValueError("Balance must be {'auto', int, dict}")
	if verbose : print("Oversampling")
	oversampled = {}
	for label in separated:
		if n_samples[label] > 0 and len(separated[label]) > 1:
			if verbose : print("\tModelling distribution for", str(label))
			model = _fit_cluster(separated[label], seed=seed)
			if verbose : print("\tSampling data for", str(label))
			model.weights_ = (model.weights_ / np.sum(model.weights_)).astype(np.float64)
			oversampled[label] = model.sample(n_samples[label])[0]
		else:
			if verbose : print("\tNo oversampling done for", str(label))
	if verbose : print("Collating and shuffling")
	new_set = []
	for k in oversampled.keys():
		length = len(oversampled[k])
		labels = np.array([k]*length).reshape(length, 1)
		new_set += list(np.concatenate((oversampled[k], labels), axis=1))
	np.random.RandomState(seed=seed).shuffle(new_set)
	new_set = np.array(new_set)
	return new_set[:,:-1], new_set[:,-1].astype(int)

def _fit_cluster(data, seed=None):
	data = np.array(data)
	models = []
	abic = []
	n_components = min([len(data), 10])
	for i in range(n_components):
		m = GMM(n_components=i+1, n_init=5, random_state=seed)
		m.fit(data)
		models.append(m)
		abic.append(np.mean([m.bic(data), m.aic(data)]))
	return models[np.argmin(abic)]
