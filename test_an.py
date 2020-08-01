import numpy as np
from pprint import pprint

from sleepens.metanetwork import ArbitratedNetwork, ArbitratedRecallNetwork, NeuralNetwork
from nnrf import NNDT, NNRF

from sleepens.analysis import classification_report, confusion_matrix
from sleepens.io import Dataset

from sleepens.utils.data import sampling as s

def holdout(arr, holdout=0.5):
	holdout = int(len(data)*holdout)
	arr1, arr2 = arr[:holdout], arr[holdout:]
	return arr1, arr2

def shuffle_ds(data, labels, seed=None):
	labels = labels.reshape(len(labels), 1)
	shuffling = np.concatenate((data, labels), axis=1)
	np.random.RandomState(seed=seed).shuffle(shuffling)
	shuffling = np.array(shuffling)
	data, labels = shuffling[:,:-1], shuffling[:,-1].astype(int)
	return data, labels

nn1 = NeuralNetwork(layers=(200,50))
nn = NeuralNetwork(layers=(400,600,100), batch_size=32, verbose=1)

an = ArbitratedNetwork(estimator=nn1, max_iter=100, batch_size=32, tol=1e-6, verbose=1)
arn = ArbitratedRecallNetwork(estimator=nn1, max_iter=100, batch_size=32, tol=1e-6, verbose=1)

files = ["/mnt/d/Documents/Research/sleepens/data/big_ds.xls"]
ds = Dataset()
ds.read(files, cols=np.arange(16), label_cols=[16])
features, data, labels = ds.features, ds.data, ds.labels

data -= np.mean(data, axis=0)
print(np.bincount(labels.reshape(-1).astype(int)))
labels = labels.reshape(-1).astype(int)
test_data, test_labels = shuffle_ds(data, labels, seed=0)

s_data, s_labels = s.multiply(data, labels, factor=0.5, verbose=1)

test_data, train_data = holdout(test_data, holdout=0.5)
test_labels, train_labels = holdout(test_labels, holdout=0.5)
train_data = np.concatenate((train_data, s_data))
train_labels = np.concatenate((train_labels, s_labels))

'''s_data, s_labels = s.balance(train_data, train_labels, verbose=1)
train_data = np.concatenate((train_data, s_data))
train_labels = np.concatenate((train_labels, s_labels))'''
print(np.bincount(train_labels.reshape(-1).astype(int)))

arn.fit(train_data, train_labels)
pred = arn.predict(test_data)
pprint(confusion_matrix(pred, test_labels))
pprint(classification_report(pred, test_labels))
