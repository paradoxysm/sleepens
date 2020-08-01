import numpy as np
from pathlib import Path
from sleepens.metanetwork import ArbitratedNetwork
from nnrf import NNRF

from sleepens import AbstractSleepEnsemble
from sleepens.ml.models import tsNeuralNetwork

from sleepens.process import primary as prim
from sleepens.process import secondary as sec
from sleepens.utils.data import sampling as s
from sleepens.utils.data import normalize as n
from sleepens.io.smrMAT import read_smrMAT
from sleepens.io import Dataset

default_params = {
	'EPOCH_SIZE': 5,
	'EEG_NAME': 'EEG',
	'NECK_NAME': 'NECK',
	'MASS_NAME': 'MASS',
	'SCORE_NAME': 'SCORE',
	'SCORE_MAP': { 'AW': 0, 'QW': 1, 'NR': 2, 'R': 3 },
	'EEG_FFT': {
					'NPERSEG': 0.75,
					'NOVERLAP': 0.5,
					'DETREND': 'constant',
				},
	'EMG_FFT': {
					'NPERSEG': 0.75,
					'NOVERLAP': 0.5,
					'DETREND': 'constant',
				},
	'PRCTILE': 95,
	'EPOCHED_VAR': {
					'SUBEPOCH_SIZE': 10,
					'THRESHOLD': 'median',
					'MERGE': 'sum'
				},
	'BANDS': {
				'DELTA' : (0.5, 4),
				'THETA' : (7, 10),
				'ALPHA'  : (11, 15),
				'BETA' : (15, 40),
			},
	'BAND_MERGE': 'sum',
	'RATIOS': {
				'THETA/DELTA': ('THETA','DELTA'),
				'ALPHA/DELTA':('ALPHA','DELTA'),
				'BETA/DELTA':('BETA','DELTA'),
				'ALPHA/THETA':('ALPHA','THETA'),
				'BETA/THETA':('BETA','THETA'),
				'BETA/ALPHA':('BETA','ALPHA')
			},
	'EMG_RMS_MERGE': 'sum',
	'EMG_PRCT_MERGE': 'sum',
	'EMG_TWITCH_MERGE': 'sum',
	'EMG_ENTROPY_MERGE': 'mean',
	'TRANSFORM': [{'FUNCTION':n.log_normalize, 'ARGS':['median', (-5, 5), 0]}]
}

class StaticSleepEnsemble(AbstractSleepEnsemble):
	def __init__(self, random_state=None, verbose=0):
		super().__init__(random_state=random_state, verbose=verbose)
		self.an = ArbitratedNetwork(random_state=random_state, verbose=verbose)
		self.tsnn = tsNeuralNetwork(random_state=random_state, verbose=verbose)
		self.params = default_params

	# Add verbosity statements!
	def fit(self, X, Y):
		s_x, s_Y = s.multiply(X, Y, factor=2,
								seed=self.random_state, verbose=self.verbose)
		self.an.fit(s_X, s_Y)
		X_, Y_ = self.an.predict_proba(X), Y
		self.tsnn.fit(X_, Y_)
		return self

	def predict_proba(self, X):
		X_ = self.an.predict_proba(X)
		pred = self.tsnn.preedict_proba(X_)
		pred = self._fix(pred)
		return pred

	def process(self, filepath, labels=False):
		eeg, neck, mass, score = self._read_data(filepath, labels=labels)
		if self.verbose > 0 : print("Extracting Features")
		eeg, neck, mass = self._detrend(eeg, neck, mass)
		eeg_fft = self._fft(self.params['EEG_FFT'], eeg)[0]
		neck_fft, mass_fft = self._fft(self.params['EMG_FFT'], neck, mass)
		eeg_rms, neck_rms, mass_rms = self._rms(eeg, neck, mass)
		neck_prct, mass_prct = self._prct(neck, mass)
		neck_twitch, mass_twitch = self._ep_var(neck, mass)
		eeg_entropy, neck_entropy, mass_entropy = self._entropy(eeg_fft, neck_fft, mass_fft)
		if self.verbose > 1 : print("Calculating Frequency Bands")
		eeg_bands = sec.spectral_band(eeg_fft, bands=self.params['BANDS'], merge=self.params['BAND_MERGE'])
		if self.verbose > 1 : print("Calculating Frequency Ratios")
		ratios = sec.ratio(eeg_bands, ratios=self.params['RATIOS'])
		if self.verbose > 1 : print("Merging")
		emg_rms = self._merge(mass_rms, neck_rms, feature='EMG RMS',
									method=self.params['EMG_RMS_MERGE'])
		emg_prct = self._merge(mass_prct, neck_prct, feature='EMG PRCT',
									method=self.params['EMG_PRCT_MERGE'])
		emg_twitch = self._merge(mass_twitch, neck_twitch, feature='EMG TWITCH',
									method=self.params['EMG_TWITCH_MERGE'])
		emg_entropy = self._merge(mass_entropy, neck_entropy, feature='EMG ENTROPY',
									method=self.params['EMG_ENTROPY_MERGE'])
		features = [eeg_bands, ratios, eeg_rms, emg_rms, emg_prct,
						eeg_entropy, emg_entropy, emg_twitch]
		ds = self._create_ds(features, overlap=False)
		if self.verbose > 1 : print("Transforming Data")
		ds = sec.transform(ds, ops=self.params['TRANSFORM'])
		if labels:
			if self.verbose > 1 : print("Adding Labels")
			score = Dataset(label_names=['SCORE'], labels=score.data[:-1].reshape(-1,1))
		ds = ds.concatenate(score)
		ds = ds.clean()
		ds.name = Path(filepath).stem
		return ds

	def export_model(self):
		raise NotImplementedError("No export_model function implemented")

	def load_model(self, filepath):
		raise NotImplementedError("No load_model function implemented")

	def _read_data(self, filepath, labels=False):
		if self.verbose > 1 : print("Reading", filepath)
		eeg = read_smrMAT(filepath, self.params['EEG_NAME'], name='EEG')
		neck = read_smrMAT(filepath, self.params['NECK_NAME'], name='NECK')
		mass = read_smrMAT(filepath, self.params['MASS_NAME'], name='MASS')
		if labels:
			labels = read_smrMAT(filepath, self.params['SCORE_NAME'], name='SCORE',
									score=True, map=self.params['SCORE_MAP'])
		return eeg, neck, mass, labels

	def _detrend(self, *channels):
		if self.verbose > 1 : print("Detrending Data")
		for c in channels:
			c.process(n.demean)
		return channels

	def _fft(self, params, *channels):
		if self.verbose > 1 : print("Conducting FFT Analysis")
		result = []
		for c in channels:
			fft = prim.fft(c, self.params['EPOCH_SIZE'],
							nperseg_factor=params['NPERSEG'],
							noverlap_factor=params['NOVERLAP'],
							detrend=params['DETREND'])
			result.append(fft)
		return result

	def _rms(self, *channels):
		if self.verbose > 1 : print("Calculating RMS")
		result = []
		for c in channels:
			rms = prim.rms(c, self.params['EPOCH_SIZE'])
			result.append(rms)
		return result

	def _prct(self, *channels):
		if self.verbose > 1 : print("Calculating Percentile Mean")
		result = []
		for c in channels:
			prct = prim.percentile_mean(c, self.params['EPOCH_SIZE'],
							k=self.params['PRCTILE'])
			result.append(prct)
		return result

	def _ep_var(self, *channels):
		if self.verbose > 1 : print("Calculating Epoched Variance")
		sub_epoch_size = self.params['EPOCHED_VAR']['SUBEPOCH_SIZE']
		threshold = self.params['EPOCHED_VAR']['THRESHOLD']
		merge = self.params['EPOCHED_VAR']['MERGE']
		result = []
		for c in channels:
			epvar = prim.epoched_variance(c, self.params['EPOCH_SIZE'],
							sub_epoch_size= sub_epoch_size,
							threshold=threshold, merge=merge)
			result.append(epvar)
		return result

	def _entropy(self, *channels):
		if self.verbose > 1 : print("Calculating Spectral Entropy")
		result = []
		for c in channels:
			entropy = sec.spectral_entropy(c)
			result.append(entropy)
		return result

	def _merge(self, channel1, channel2, feature='MERGE', method='mean'):
		if method is None:
			raise ValueError("Feature")
		merge = channel1.concatenate(channel2, overlap=False)
		return sec.merge(merge, feature=feature, method=method)

	def _create_ds(self, features, overlap=False):
		if self.verbose > 1 : print("Creating Dataset")
		if len(features) == 0 : return Dataset()
		ds = features.pop(0)
		for f in features:
			ds.concatenate(f, overlap=overlap)
		return ds

	def _fix(X):
		pass
