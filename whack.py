import numpy as np
from pprint import pprint

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, make_scorer, recall_score

from sleep.utils.analysis import classification_report, confusion_matrix
from sleep.utils.data import sampling as s
from sleep.io import Dataset
from sleep.utils.misc import separate_by_label
from sleep.ml.linear import PCA

def ovr_label(labels, state):
	ovr = np.where(labels == state, 1, 0)
	return ovr

def ovo_label(labels, s1, s2):
	ovo = np.where(labels == s1, 1, 0)
	ovo = np.where(labels == s2, 2, ovo)
	return ovo

def shuffle_ds(data, labels, seed=None):
	labels = labels.reshape(len(labels), 1)
	shuffling = np.concatenate((data, labels), axis=1)
	np.random.RandomState(seed=seed).shuffle(shuffling)
	shuffling = np.array(shuffling)
	data, labels = shuffling[:,:-1], shuffling[:,-1].astype(int)
	return data, labels

def holdout(arr, holdout=0.5):
	holdout = int(len(data)*holdout)
	arr1, arr2 = arr[:holdout], arr[holdout:]
	return arr1, arr2

def create_model():
	return RandomForestClassifier(n_estimators=150, criterion='entropy', max_features='auto', random_state=0)

def print_feature_importances(model, features):
	print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), features),
             reverse=True))

files = ["/mnt/c/Users/jfw20/Desktop/Lab/sleep/data/bigboi.xls"]
ds = Dataset()
ds.read(files, cols=np.arange(32), label_cols=[32])
features, data, labels = ds.features, ds.data, ds.labels
data -= np.mean(data, axis=0)
#pc_data = PCA().fit(data)
#pc_features = np.arange(14)
#data = np.concatenate((data, pc_data), axis=1)
#features = np.concatenate((features, pc_features), axis=0)
print(np.bincount(labels.reshape(-1).astype(int)))
labels = labels.reshape(-1).astype(int)
data, labels = shuffle_ds(data, labels, seed=0)
train_data, test_data = holdout(data)
train_labels, test_labels = holdout(labels)
print(np.bincount(test_labels))

aw_labels, qw_labels = ovr_label(labels, 0), ovr_label(labels, 1)
nr_labels, r_labels = ovr_label(labels, 2), ovr_label(labels, 3)
aw_train, aw_test = holdout(aw_labels)
qw_train, qw_test = holdout(qw_labels)
nr_train, nr_test = holdout(nr_labels)
r_train, r_test = holdout(r_labels)
qw_data, qw_lbl = s.generative_oversample(train_data, train_labels, balance={0:1000,1:4000,2:1000,3:1000}, seed=0)
r_data, r_lbl = s.generative_oversample(train_data, train_labels, balance={0:1000,1:1000,2:1000,3:4000}, seed=0)
qw_lbl, r_lbl = ovr_label(qw_lbl, 1), ovr_label(r_lbl, 3)

aw, qw, nr, r = create_model(), create_model(), create_model(), create_model()

print("Fitting base classifiers")
aw.fit(train_data, aw_train)
qw.fit(qw_data, qw_lbl)
nr.fit(train_data, nr_train)
r.fit(r_data, r_lbl)
print("Predicting")
aw_prob = aw.predict_proba(test_data)[:,-1]
aw_pred = aw.predict(test_data)
qw_prob = qw.predict_proba(test_data)[:,-1]
qw_pred = qw.predict(test_data)
nr_prob = nr.predict_proba(test_data)[:,-1]
nr_pred = nr.predict(test_data)
r_prob = r.predict_proba(test_data)[:,-1]
r_pred = r.predict(test_data)

print('AW')
pprint(confusion_matrix(aw_pred, test_labels))
print_feature_importances(aw, features)
print('QW')
pprint(confusion_matrix(qw_pred, test_labels))
print_feature_importances(qw, features)
print('NR')
pprint(confusion_matrix(nr_pred, test_labels))
print_feature_importances(nr, features)
print('R')
pprint(confusion_matrix(r_pred, test_labels))
print_feature_importances(r, features)

print("Setting up conflict data")
aw_qw_conf = np.where(np.logical_and(aw_pred == 1, qw_pred == 1))[0]
aw_nr_conf = np.where(np.logical_and(aw_pred == 1, nr_pred == 1))[0]
aw_r_conf = np.where(np.logical_and(aw_pred == 1, r_pred == 1))[0]
qw_nr_conf = np.where(np.logical_and(qw_pred == 1, nr_pred == 1))[0]
qw_r_conf = np.where(np.logical_and(qw_pred == 1, r_pred == 1))[0]
nr_r_conf = np.where(np.logical_and(nr_pred == 1, r_pred == 1))[0]
aw_qw_data = np.concatenate((test_data, aw_prob.reshape(-1,1), qw_prob.reshape(-1,1)), axis=1)
aw_nr_data = np.concatenate((test_data, aw_prob.reshape(-1,1), nr_prob.reshape(-1,1)), axis=1)
aw_r_data = np.concatenate((test_data, aw_prob.reshape(-1,1), r_prob.reshape(-1,1)), axis=1)
qw_nr_data = np.concatenate((test_data, qw_prob.reshape(-1,1), nr_prob.reshape(-1,1)), axis=1)
qw_r_data = np.concatenate((test_data, qw_prob.reshape(-1,1), r_prob.reshape(-1,1)), axis=1)
nr_r_data = np.concatenate((test_data, nr_prob.reshape(-1,1), r_prob.reshape(-1,1)), axis=1)
aw_qw_data, aw_qw_labels = aw_qw_data[aw_qw_conf], ovo_label(test_labels, 0, 1)[aw_qw_conf]
aw_nr_data, aw_nr_labels = aw_nr_data[aw_nr_conf], ovo_label(test_labels, 0, 2)[aw_nr_conf]
aw_r_data, aw_r_labels = aw_r_data[aw_r_conf], ovo_label(test_labels, 0, 3)[aw_r_conf]
qw_nr_data, qw_nr_labels = qw_nr_data[qw_nr_conf], ovo_label(test_labels, 1, 2)[qw_nr_conf]
qw_r_data, qw_r_labels = qw_r_data[qw_r_conf], ovo_label(test_labels, 1, 3)[qw_r_conf]
nr_r_data, nr_r_labels = nr_r_data[nr_r_conf], ovo_label(test_labels, 2, 3)[nr_r_conf]
awqw_data, awqw_labels = s.generative_oversample(aw_qw_data, aw_qw_labels, balance='f20', seed=0)
awnr_data, awnr_labels = s.generative_oversample(aw_nr_data, aw_nr_labels, balance='f20', seed=0)
awr_data, awr_labels = s.generative_oversample(aw_r_data, aw_r_labels, balance='f20', seed=0)
qwnr_data, qwnr_labels = s.generative_oversample(qw_nr_data, qw_nr_labels, balance='f20', seed=0)
qwr_data, qwr_labels = s.generative_oversample(qw_r_data, qw_r_labels, balance='f20', seed=0)
nrr_data, nrr_labels = s.generative_oversample(nr_r_data, nr_r_labels, balance='f20', seed=0)

aw_qw, aw_nr, aw_r = create_model(), create_model(), create_model()
qw_nr, qw_r, nr_r = create_model(), create_model(), create_model()

print("Fitting arbitrators")
if len(awqw_data) > 0: aw_qw.fit(awqw_data, awqw_labels)
if len(awnr_data) > 0: aw_nr.fit(awnr_data, awnr_labels)
if len(awr_data) > 0: aw_r.fit(awr_data, awr_labels)
if len(qwnr_data) > 0: qw_nr.fit(qwnr_data, qwnr_labels)
if len(qwr_data) > 0: qw_r.fit(qwr_data, qwr_labels)
if len(nrr_data) > 0: nr_r.fit(nrr_data, nrr_labels)
print("Predicting")
if len(aw_qw_data) > 0:
	awqw_prob, awqw_pred = aw_qw.predict_proba(aw_qw_data), aw_qw.predict(aw_qw_data)
	print('AW-QW')
	pprint(confusion_matrix(awqw_pred, test_labels[aw_qw_conf]))
	print_feature_importances(aw_qw, features)
else: awqw_prob, awqw_pred = np.array([]), np.array([])
if len(aw_nr_data) > 0:
	awnr_prob, awnr_pred = aw_nr.predict_proba(aw_nr_data), aw_nr.predict(aw_nr_data)
	print('AW-NR')
	pprint(confusion_matrix(awnr_pred, test_labels[aw_nr_conf]))
	print_feature_importances(aw_nr, features)
else: awnr_prob, awnr_pred = np.array([]), np.array([])
if len(aw_r_data) > 0:
	awr_prob, awr_pred = aw_r.predict_proba(aw_r_data), aw_r.predict(aw_r_data)
	print('AW-R')
	pprint(confusion_matrix(awr_pred, test_labels[aw_r_conf]))
	print_feature_importances(aw_r, features)
else: awr_prob, awr_pred = np.array([]), np.array([])
if len(qw_nr_data) > 0:
	qwnr_prob, qwnr_pred = qw_nr.predict_proba(qw_nr_data), qw_nr.predict(qw_nr_data)
	print('QW-NR')
	pprint(confusion_matrix(qwnr_pred, test_labels[qw_nr_conf]))
	print_feature_importances(qw_nr, features)
else: qwnr_prob, qwnr_pred = np.array([]), np.array([])
if len(qw_r_data) > 0:
	qwr_prob, qwr_pred = qw_r.predict_proba(qw_r_data), qw_r.predict(qw_r_data)
	print('QW-R')
	pprint(confusion_matrix(qwr_pred, test_labels[qw_r_conf]))
	print_feature_importances(qw_r, features)
else: qwr_prob, qwr_pred = np.array([]), np.array([])
if len(nr_r_data) > 0:
	nrr_prob, nrr_pred = nr_r.predict_proba(nr_r_data), nr_r.predict(nr_r_data)
	print('NR-R')
	pprint(confusion_matrix(nrr_pred, test_labels[nr_r_conf]))
	print_feature_importances(nr_r, features)
else: nrr_prob, nrr_pred = np.array([]), np.array([])

print("Aggregating decisions")
'''
Should decisions be determined by simple majority vote of abritrators?
Or by collective confidence votes of arbitrators? (include base classifiers?)
'''
one_hot = np.concatenate((aw_pred.reshape(-1,1),qw_pred.reshape(-1,1),nr_pred.reshape(-1,1),r_pred.reshape(-1,1)),axis=1)
for i in range(len(awqw_pred)):
	if awqw_pred[i] == 1 : one_hot[aw_qw_conf[i], 0] += 1
	elif awqw_pred[i] == 2 : one_hot[aw_qw_conf[i], 1] += 1
	else : one_hot[aw_qw_conf[i], [0,1]] -= 1
for i in range(len(awnr_pred)):
	if awnr_pred[i] == 1 : one_hot[aw_nr_conf[i], 0] += 1
	elif awnr_pred[i] == 2 : one_hot[aw_nr_conf[i], 2] += 1
	else : one_hot[aw_nr_conf[i], [0,2]] -= 1
for i in range(len(awr_pred)):
	if awr_pred[i] == 1 : one_hot[aw_r_conf[i], 0] += 1
	elif awr_pred[i] == 2 : one_hot[aw_r_conf[i], 3] += 1
	else : one_hot[aw_r_conf[i], [0,3]] -= 1
for i in range(len(qwnr_pred)):
	if qwnr_pred[i] == 1 : one_hot[qw_nr_conf[i], 1] += 1
	elif qwnr_pred[i] == 2 : one_hot[qw_nr_conf[i], 2] += 1
	else : one_hot[qw_nr_conf[i], [1,2]] -= 1
for i in range(len(qwr_pred)):
	if qwr_pred[i] == 1 : one_hot[qw_r_conf[i], 1] += 1
	elif qwr_pred[i] == 2 : one_hot[qw_r_conf[i], 3] += 1
	else : one_hot[qw_r_conf[i], [1,3]] -= 1
for i in range(len(nrr_pred)):
	if nrr_pred[i] == 1 : one_hot[nr_r_conf[i], 2] += 1
	elif nrr_pred[i] == 2 : one_hot[nr_r_conf[i], 3] += 1
	else : one_hot[nr_r_conf[i], [2,3]] -= 1

print("Fitting trash compactor")
one_hot = np.nan_to_num(one_hot / one_hot.max(axis=1).reshape(-1,1)).astype(int)
trash_idx = np.where(one_hot.sum(axis=1) != 1)
trash_data, trash_labels = test_data[trash_idx], test_labels[trash_idx]
t_data, t_labels = s.generative_oversample(trash_data, trash_labels, balance='f20', seed=0)
trash = create_model()
trash.fit(t_data, t_labels)
t_pred = trash.predict(trash_data)
print("TRASH")
pprint(confusion_matrix(t_pred, trash_labels))
print_feature_importances(trash, features)
prediction = np.argmax(one_hot, axis=1)
print("ORIGINAL PRED")
pprint(confusion_matrix(prediction, test_labels))
prediction[trash_idx] = t_pred
print("FINAL")
pprint(confusion_matrix(prediction, test_labels))

with open("labels.txt", 'w') as file:
	for p in prediction:
		file.write(str(p) + "\n")

with open("score.txt", 'w') as file:
	for p in test_labels:
		file.write(str(p) + "\n")
