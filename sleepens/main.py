import tkinter as tk
from tkinter import filedialog
import os, joblib, importlib, time, sys, struct
import numpy as np
from tqdm import trange
from pathlib import Path
from pprint import pprint

from sleepens.io import Dataset, DataObject
from sleepens.ml import cross_validate
from sleepens.analysis import confusion_matrix, classification_report
from sleepens.io.interfaces import interfaces

from sleepens.protocols import protocols
from sleepens.protocols.sleepens4 import SleepEnsemble4

version = "0.1.0"
model = SleepEnsemble4()
verbose = 4

def displayMenu():
	print("-"*30)
	print("Sleep Ensemble", version)
	print("Current model:", model.name)
	print("-"*30)
	print("1. Classify some files")
	print("2. Train a new Sleep Ensemble")
	print("3. Validate model")
	print("4. Load model")
	print("5. Export current model")
	print("6. Change verbosity of messages (currently", str(verbose) + ")")
	print("7. Update parameters")
	print("Q. Exit (Emergency abort is Ctrl+C)")
	print("-"*30)

def yes_no_loop(msg):
	while True:
		usr = input(msg + " [y/n]: ").upper()
		if usr == "Y" : return True
		elif usr== "N" : return False
		else : print("Sorry, didn't quite catch that.")

def import_module(name, path):
	try:
		spec = importlib.util.spec_from_file(name, path)
		mod = util.module_from_spec(spec)
		spec.loader.exec_module(mod)
		return mod
	except:
		return importlib.machinery.SourceFileLoader(name, path).load_module()

def ask_filename(title="Choose File", filetypes=[("Any File", "*")]):
	root = tk.Tk()
	root.withdraw()
	filepath = filedialog.askopenfilename(title=title,
					filetypes=filetypes)
	root.destroy()
	return filepath

def ask_filenames(title="Choose Files", filetypes=[("Any File", "*")]):
	root = tk.Tk()
	root.withdraw()
	filepaths = root.tk.splitlist(filedialog.askopenfilenames(title=title,
					filetypes=filetypes))
	root.destroy()
	return filepaths

def save_file(title="Save File", defaultextension=".txt", filetypes=[("Any File", "*")]):
	root = tk.Tk()
	root.withdraw()
	file = filedialog.asksaveasfilename(title=title, filetypes=filetypes)
	if not file : return file
	version = "-py"
	for i in range(3):
		version += str(sys.version_info[i]) + "."
	version += str(sys.hexversion) + "-"
	file += version + sys.platform + "-" + str(struct.calcsize("P")*8) + "bit"
	file += defaultextension
	root.destroy()
	return file

def ask_directory(title="Choose Folder"):
	root = tk.Tk()
	root.withdraw()
	folder = filedialog.askdirectory(title=title)
	root.destroy()
	return folder

def print_report(Y_hat, Y):
	print("Confusion Matrix")
	print("-"*30)
	matrix, targets, labels = confusion_matrix(Y_hat, Y)
	row_format = "{:>10}" * (len(labels) + 2)
	title = ["" for i in range(len(labels)+2)]
	title[len(labels)//2 + 2] = "PREDICTION"
	print(row_format.format(*title))
	print(row_format.format("", "", *labels))
	for row in range(len(matrix)):
		if row == len(matrix)//2 : blank = "TARGET"
		else : blank = ""
		row_data = ["%.f" % cell for cell in matrix[row]]
		print(row_format.format(blank, "%.f" % targets[row], *row_data))
	print("-"*30)
	print("Classification Report")
	print("-"*30)
	report = classification_report(Y_hat, Y)
	beta = report.pop('beta')
	support = report.pop('support')
	accuracy = report.pop('accuracy')
	macro = report.pop('macro avg')
	weighted = report.pop('weighted avg')
	header = ["Precision", "Recall", "F1-Score", "Support"]
	row_format = "{:>15}" * (len(header) + 1)
	print(row_format.format("", *header))
	for k, v in sorted(report.items()):
		print(row_format.format("%.f" % k, "%.4f" % v['precision'], "%.4f" % v['recall'],
								"%.4f" % v['f-score'], "%.f" % v['support']))
	print("")
	print(row_format.format('accuracy', "", "", "%.4f" % accuracy, "%.f" % support))
	print(row_format.format('macro avg', "%.4f" % macro['precision'], "%.4f" % macro['recall'],
							"%.4f" % macro['f-score'], "%.f" % support))
	print(row_format.format('weighted avg', "%.4f" % weighted['precision'], "%.4f" % weighted['recall'],
							"%.4f" % weighted['f-score'], "%.f" % support))
	print("-"*30)

def change_protocol():
	global model, verbose
	while True:
		print("-"*30)
		print("Select Sleep Ensemble Protocol")
		print("-"*30)
		for i in range(len(protocols)):
			print(str(i+1)+".", protocols[i].__name__)
		print("Q. Return to main menu")
		print("-"*30)
		usr = input("Select protocol: ").upper()
		if usr.isdigit():
			usr = int(usr)
			if usr > 0 and usr <= len(protocols):
				protocol = protocols[usr-1]
				print("Changed protocol to", protocol.__name__)
				return protocol
			else : print("Sorry, didn't quite catch that.")
		elif usr == "Q": return
		else : print("Sorry, didn't quite catch that.")

def read_data(in_int, filepath, labels=False):
	global model, verbose
	if verbose > 1 : print("Reading file", filepath)
	if in_int.type == "RAW":
		data, labels = model.read(in_int, filepath, labels=labels)
		name = Path(filepath).stem
		ds = model.process(data, labels, name)
	elif in_int.type == "DATASET":
		ds = in_int.read_data(filepath, model.params['reader']['DATA_COLS'])
		if labels:
			label_ds = in_int.read_labels(filepath, model.params['reader']['LABEL_COLS'])
			ds = ds.concatenate(label_ds)
		data, labels = None, label_ds.labels
	else:
		print("Could not read file due to bad interface")
		return None
	return ds, data, labels

def write_data(out_int, destination, name, data, ds, p=None, save_input=True):
	global model, verbose
	if verbose > 1 : print("Writing results")
	if p is None:
		filepath = destination + '/' + name + '-train' + out_int.filetypes[0][1][1:]
	else:
		filepath = destination + '/' + name + '-predictions' + out_int.filetypes[0][1][1:]
	if out_int.type == "RAW":
		if p is None:
			print("No predictions used, no need to write raw data again")
			return
		result = DataObject(name="P", data=p, resolution=data[0].resolution/data[0].divide)
		out_int.write(filepath + 'x', data, result, map=model.params['process']['SCORE_MAP'],
						epoch_size=model.params['process']['EPOCH_SIZE'])
	elif out_int.type == "DATASET":
		if p is None and not save_input:
			return
		elif p is None:
			result = ds
		elif p is not None:
			result = Dataset(label_names=['P'], labels=p)
			if save_input : result = ds.concatenate(result)
		out_int.write(filepath, result)
	else:
		print("Could not write results due to bad interface")

def classify():
	global model, verbose
	print("-"*30)
	print("Sleep Ensemble Classification")
	print("Current model:", model.name)
	print("-"*30)
	print("Select input interface")
	in_int = select_interface(tags={'r'})
	if in_int is None:
		print("You didn't select an interface! Returning you to the main menu")
		return
	print("Select files to classify")
	print("Currently configured to accept", in_int.standard)
	filepaths = ask_filenames(filetypes=in_int.filetypes)
	if len(filepaths) == 0:
		print("You didn't select any files! Returning you to the main menu")
		return
	print("Select where classifications should go:")
	destination = ask_directory()
	if destination is None or not os.path.isdir(destination):
		print("You didn't select a destination! Returning you to the main menu")
		return
	print("Select output interface")
	out_int = select_interface(tags={'w'})
	if out_int is None:
		print("You didn't select an interface! Returning you to the main menu")
		return
	save_input = yes_no_loop("Do you also want to export processed data alongside classifications?")
	print("Identified", len(filepaths), "files to classify")
	if verbose == 1 : jobs = trange(len(filepaths))
	else : jobs = range(len(filepaths))
	for i in jobs:
		if verbose > 1 : print("Classifying", filepaths[i])
		ds, data, labels = read_data(in_int, filepaths[i], labels=False)
		p, y_hat = model.predict([ds.data])
		write_data(out_int, destination, ds.name, data, ds, p=p[0], save_input=save_input)
	print("Completed classification jobs!")

def train():
	global model, verbose
	print("-"*30)
	print("Sleep Ensemble Training")
	print("Current model:", model.name)
	print("-"*30)
	print("1. Create a new model from a protocol class")
	print("2. Train from the current model")
	print("Q. Return to main menu")
	print("-"*30)
	usr = input("Select an option: ").upper()
	if usr.isdigit():
		usr = int(usr)
		if usr == 1:
			change_protocol()
		elif usr == 2:
			warm_start = yes_no_loop("Do you want a warm start with this model?")
			model.classifier.set_warm_start(warm_start)
		else : print("Sorry, didn't quite catch that.")
	elif usr == "Q": return
	else : print("Sorry, didn't quite catch that.")
	print("Training from model:", model.name)
	print("Select input interface")
	in_int = select_interface(tags={'r'})
	if in_int is None:
		print("You didn't select an interface! Returning you to the main menu")
		return
	print("Select files to use for training")
	print("Currently configured to accept", in_int.standard)
	filepaths = ask_filenames(filetypes=in_int.filetypes)
	if len(filepaths) == 0:
		print("You didn't select any files! Returning you to the main menu")
		return
	print("Identified", len(filepaths), "files for training")
	ds, data, labels = [], [], []
	if verbose == 1 : jobs = trange(len(filepaths))
	else : jobs = range(len(filepaths))
	for i in jobs:
		ds_, _, _ = read_data(in_int, filepaths[i], labels=True)
		ds.append(ds_)
		data.append(ds_.data)
		labels.append(ds_.labels)
	print("Training model on", len(np.concatenate(data)), "samples")
	model.fit(data, labels)
	print("Completed Training!")
	save_input = yes_no_loop("Do you want to export processed training data?")
	if save_input:
		print("Select output interface")
		out_int = select_interface(tags={'w'})
		if out_int is None:
			print("You didn't select an interface! Returning you to the main menu")
			return
		print("Select where data should be saved")
		destination = ask_directory()
		if destination is None or not os.path.isdir(destination):
			print("You didn't select a destination! Returning you to the main menu")
			return
		for ds_ in ds:
			write_data(out_int, destination, ds_.name, ds_.data, ds_)
	print("Complete! You can export model from the main menu")

def validate():
	global model, verbose
	print("-"*30)
	print("Sleep Ensemble Validation")
	print("Current model:", model.name)
	print("-"*30)
	print("Select input interface")
	in_int = select_interface(tags={'r'})
	if in_int is None:
		print("You didn't select an interface! Returning you to the main menu")
		return
	print("Select files to use for validation:")
	filepaths = ask_filenames(filetypes=in_int.filetypes)
	if len(filepaths) == 0:
		print("You didn't select any files! Returning you to the main menu")
		return
	save_input = yes_no_loop("Do you want to export the results?")
	if save_input:
		print("Select where results should go:")
		destination = ask_directory()
		if destination is None or not os.path.isdir(destination):
			print("You didn't select a destination! Returning you to the main menu")
			return
	print("Identified", len(filepaths), "files for validation")
	req_train = yes_no_loop("Do you wish to train and validate via cross-validation (y) or just validate (n)?")
	ds, data, labels = [], [], []
	raw_data = []
	if verbose == 1 : jobs = trange(len(filepaths))
	else : jobs = range(len(filepaths))
	for i in jobs:
		ds_, data_, labels_ = read_data(in_int, filepaths[i], labels=True)
		ds.append(ds_)
		data.append(ds_.data)
		labels.append(ds_.labels)
		raw_data.append(data_)
	if req_train:
		p, Y_hat = model.cross_validate(data, labels)
	else:
		p, Y_hat = model.predict(data)
	if p[0].ndim != 1:
		p_ = []
		start = 0
		for i in range(len(data)):
			p_.append(p[start:start + len(data[i])])
			start = start + len(data[i])
		p = p_
	score = model.score(p, labels)
	p_overall = np.concatenate(p)
	labels_overall = np.concatenate(labels)
	print_report(p_overall.reshape(-1), labels_overall.reshape(-1))
	if save_input:
		print("Select output interface")
		out_int = select_interface(tags={'w'})
		if out_int is None:
			print("You didn't select an interface! Returning you to the main menu")
			return
		for i in range(len(p)):
			write_data(out_int, destination, ds[i].name + "-%.4f" % score[i], raw_data[i], ds[i], p=p[i])
	print("Completed validation")

def load():
	global model, verbose
	print("-"*30)
	print("Load Model")
	print("Current model:", model.name)
	print("-"*30)
	change_protocol()
	file = ask_filename(title="Choose Model File", filetypes=[("Joblib", "*.joblib")])
	if not file:
		print("You didn't select a file! Returning you to the main menu")
		return
	print("Selected", Path(file).stem)
	confirm = yes_no_loop("Are you sure? Only load .joblib files from trusted sources")
	if not confirm:
		print("Cancelling import. Returning you to the main menu")
		return
	model = joblib.load(file)
	model.set_verbose(verbose - 1)
	print("Loaded", model.name)

def export():
	global model, verbose
	print("-"*30)
	print("Export Model")
	print("Current model:", model.name)
	print("-"*30)
	name_change = yes_no_loop("Do you want to change the name of the model?")
	if name_change:
		model.name = input("Provide a name:")
	file = save_file(title="Save Model File", defaultextension=".joblib",
					filetypes=[("Joblib", "*.joblib")])
	if not file:
		print("Cancelling save. Returning you to the main menu")
		return
	print("Exporting", model.name, "to", Path(file).stem)
	joblib.dump(model, file)
	print("Exported", model.name, "to", Path(file).stem)

def select_interface(tags={'r', 'w'}):
	global model, verbose
	while True:
		print("-"*30)
		print("Select File Interface")
		print("-"*30)
		accept = []
		for i in range(len(interfaces)):
			if tags.issubset(interfaces[i].tags):
				accept.append(interfaces[i])
		for i in range(len(accept)):
			print(str(i+1)+".", accept[i].name + ":", accept[i].standard, accept[i].tags)
		print("Q. Return to main menu")
		print("-"*30)
		usr = input("Select new file interface: ").upper()
		if usr.isdigit():
			usr = int(usr)
			if usr > 0 and usr <= len(accept):
				interface = accept[usr-1]
				print("Changed interface to", interface.name)
				return interface
			else : print("Sorry, didn't quite catch that.")
		elif usr == "Q": return
		else : print("Sorry, didn't quite catch that.")

def change_verbosity():
	global model, verbose
	while True:
		print("-"*30)
		print("Change Verbosity")
		print("Current verbosity:", verbose)
		print("-"*30)
		print("0. Silent (other than application messages)")
		print("1. Quiet - File level - Quiet")
		print("2. Quiet - File level")
		print("3. Soft - Model level")
		print("4. Soft - Classifier level - Quiet")
		print("5. Normal - Classifier level - Soft")
		print("6. Normal - Classifier level - Normal")
		print("7. Loud - Classifier level - Loud")
		print("Q. Return to main menu")
		print("-"*30)
		usr = input("Select an option: ").upper()
		if usr.isdigit():
			usr = int(usr)
			if usr >= 0 and usr <= 7:
				verbose = usr
				model.set_verbose(verbose - 1)
				print("Changed verbosity to", usr)
				return
			else : print("Sorry, didn't quite catch that.")
		elif usr == "Q": return
		else : print("Sorry, didn't quite catch that.")

def update_params():
	global model, verbose
	print("-"*30)
	print("Update Parameters")
	print("-"*30)
	pprint(model.params)
	print("-"*30)
	change = yes_no_loop("Do you wish to change parameter file?")
	if change:
		file = ask_filename(title="Choose Parameter File",
						filetypes=[("Python files", "*.py")])
		if not file:
			print("You didn't select a file! Returning you to the main menu")
			return
		print("Selected", Path(file).stem)
		confirm = yes_no_loop("Are you sure? Only load .py files from trusted sources")
		if not confirm:
			print("Cancelling import. Returning you to the main menu")
			return
		params = import_module('params', file)
		if not isinstance(params, dict):
			print("Imported something that isn't in a parameter dictionary! Returning you to the main menu")
			return
		print("Found the following:")
		pprint(params)
		finalize = yes_no_loop("Confirm update?")
		if finalize:
			model.params = params
			print("Updated parameters")
	print("Returning you to the main menu")

def main():
	print("Starting...")
	model.set_verbose(verbose - 1)
	while True:
		time.sleep(0.5)
		displayMenu()
		cmd = input("What would you like to do? ").upper()
		if cmd == "1":
			classify()
		elif cmd == "2":
			train()
		elif cmd == "3":
			validate()
		elif cmd == "4":
			load()
		elif cmd == "5":
			export()
		elif cmd == "6":
			change_verbosity()
		elif cmd == "7":
			update_params()
		elif cmd == "Q":
			print("Goodbye!")
			exit()
		else:
			print("Sorry, didn't quite catch that.")
