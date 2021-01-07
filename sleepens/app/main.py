import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
from tqdm import trange
from pathlib import Path
from pprint import pprint
import importlib
import time

from sleepens.io import Dataset
from sleepens.app._base_sleepens import check_sleepens, ShellSleepEnsemble, AbstractSleepEnsemble
from sleepens.ml import cross_validate
from sleepens.analysis import get_metrics, classification_report

from sleepens.app.sleepens4 import SleepEnsemble4

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
	print("6. Change accepted data file type (currently", model.reader.standard + ")")
	print("7. Change verbosity of messages (currently", str(verbose) + ")")
	print("8. Update parameters")
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
	file = filedialog.asksaveasfilename(title=title,
					defaultextension=defaultextension, filetypes=filetypes)
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

def classify():
	global model, verbose
	print("-"*30)
	print("Sleep Ensemble Classification")
	print("Current model:", model.name)
	print("-"*30)
	print("Select files to classify")
	print("This model is configured to accept", model.reader.standard)
	filepaths = ask_filenames(filetypes=model.reader.filetypes)
	if len(filepaths) == 0:
		print("You didn't select any files! Returning you to the main menu")
		return
	print("Select where classifications should go:")
	destination = ask_directory()
	if destination is None or not os.path.isdir(destination):
		print("You didn't select a destination! Returning you to the main menu")
		return
	save_input = yes_no_loop("Do you also want to export processed data alongside classifications?")
	print("Identified", len(filepaths), "files to classify")
	if verbose == 1 : jobs = trange(len(filepaths))
	else : jobs = range(len(filepaths))
	for i in jobs:
		if verbose > 1 : print("Classifying", filepaths[i])
		data, _ = model.read(filepaths[i], labels=False)
		name = Path(filepaths[i]).stem
		ds = model.process(data, None, name)
		p = model.predict(ds.data)
		result = Dataset(label_names=['PREDICTION'], labels=p.reshape(-1,1))
		if save_input:
			result = ds.concatenate(result)
		result.name = name + "-predictions"
		if verbose > 1 : print("Writing results")
		result.write(destination)
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
			file = ask_filename(title="Choose Model Class File",
							filetypes=[("Python files", "*.py")])
			if not file:
				print("You didn't select a file! Returning you to the main menu")
				return
			print("Selected", Path(file).stem)
			confirm = yes_no_loop("Are you sure? Only load .py files from trusted sources")
			if not confirm:
				print("Cancelling import. Returning you to the main menu")
				return
			name = input("Enter the exact name of the class to import from the .py file")
			mod = getattr(import_module(name, file), name)
			if not issubclass(mod, AbstractSleepEnsemble):
				print("Imported something that isn't a Sleep Ensemble! Returning you to the main menu")
				return
			mod = mod(verbose=model.verbose)
			print("Found the following:", mod)
			finalize = yes_no_loop("Confirm use?")
			if finalize:
				model = mod
		elif usr == 2:
			warm_start = yes_no_loop("Do you want a warm start with this model?")
			model.classifier.warm_start = warm_start
		else : print("Sorry, didn't quite catch that.")
	elif usr == "Q": return
	else : print("Sorry, didn't quite catch that.")
	print("Training from model:", model.name)
	print("Select files to use for training")
	print("This model is configured to accept", model.reader.standard)
	filepaths = ask_filenames(filetypes=model.reader.filetypes)
	if len(filepaths) == 0:
		print("You didn't select any files! Returning you to the main menu")
		return
	print("Identified", len(filepaths), "files for training")
	ds, data, labels = [], [], []
	if verbose == 1 : jobs = trange(len(filepaths))
	else : jobs = range(len(filepaths))
	for i in jobs:
		if verbose > 1 : print("Reading", filepaths[i])
		d, l = model.read(filepaths[i], labels=True)
		name = Path(filepaths[i]).stem
		ds_ = model.process(d, l, name)
		ds.append(ds_)
		data.append(ds_.data)
		labels.append(ds_.labels)
	print("Training model on", len(np.concatenate(data)), "samples")
	model.fit(data, labels)
	print("Completed Training!")
	save_input = yes_no_loop("Do you want to export processed training data?")
	if save_input:
		print("Select where data should be saved")
		destination = ask_directory()
		if destination is None or not os.path.isdir(destination):
			print("You didn't select a destination! Returning you to the main menu")
			return
		if verbose > 1 : print("Writing results")
		for ds_ in ds:
			ds_.write(destination)
	print("Complete! You can export model from the main menu")

def validate():
	global model, verbose
	print("-"*30)
	print("Sleep Ensemble Validation")
	print("Current model:", model.name)
	print("-"*30)
	print("Select files to use for validation:")
	print("This model is configured to accept", model.reader.standard)
	filepaths = ask_filenames(filetypes=model.reader.filetypes)
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
	if verbose == 1 : jobs = trange(len(filepaths))
	else : jobs = range(len(filepaths))
	for i in jobs:
		if verbose > 1 : print("Reading", filepaths[i])
		d, l = model.read(filepaths[i], labels=True)
		name = Path(filepaths[i]).stem
		ds_ = model.process(d, l, name)
		ds.append(ds_)
		data.append(ds_.data)
		labels.append(ds_.labels)
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
	print("Overall Score:", score)
	p_overall = np.concatenate(p)
	labels_overall = np.concatenate(labels)
	print_report(p_overall.reshape(-1), labels_overall.reshape(-1))
	if save_input:
		if verbose > 1 : print("Writing results")
		for i in range(len(p)):
			r = np.concatenate((Y_hat[i], p[i].reshape(-1,1)), axis=1)
			result = Dataset(label_names=['AW','QW','NR','R','P'], labels=r)
			result = ds[i].concatenate(result)
			result.name = ds[i].name + "-validated-" + "%.4f" % score[i]
			result.write(destination)
	print("Completed validation")

def load():
	global model, verbose
	print("-"*30)
	print("Load Model")
	print("Current model:", model.name)
	print("-"*30)
	file = ask_filename(title="Choose Model File", filetypes=[("Joblib", "*.joblib")])
	if not file:
		print("You didn't select a file! Returning you to the main menu")
		return
	print("Selected", Path(file).stem)
	confirm = yes_no_loop("Are you sure? Only load .joblib files from trusted sources")
	if not confirm:
		print("Cancelling import. Returning you to the main menu")
		return
	shell = ShellSleepEnsemble()
	model = shell.load(file)
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
	model.export(file)
	print("Exported", model.name, "to", Path(file).stem)

def change_reader():
	global model, verbose
	from sleepens.io.reader import readers
	while True:
		print("-"*30)
		print("Change File Reader")
		print("Current reader:", model.reader.name)
		print("-"*30)
		for i in range(len(readers)):
			print(str(i)+".", readers[i][1] + ":", readers[i][2])
		print("Q. Return to main menu")
		print("-"*30)
		usr = input("Select new file reader: ").upper()
		if usr.isdigit():
			usr = int(usr)
			if usr > 0 and usr < len(readers):
				model.reader = readers[usr][0](verbose=model.verbose)
				print("Changed reader to", readers[i][1])
				return
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
		print("0. Silent (other than main messages)")
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
	#import model
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
			change_reader()
		elif cmd == "7":
			change_verbosity()
		elif cmd == "8":
			update_params()
		elif cmd == "Q":
			print("Goodbye!")
			exit()
		else:
			print("Sorry, didn't quite catch that.")
