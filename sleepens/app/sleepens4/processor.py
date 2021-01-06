import numpy as np

from sleepens.process._base_processor import AbstractProcessor

from sleepens.process import primary as prim
from sleepens.process import secondary as sec
from sleepens.utils.data import sampling as s
from sleepens.utils.data import normalize as n
from sleepens.io.reader import smrMATReader
from sleepens.io import Dataset

from sleepens.app.sleepens4.params import params

class Sleepens4_Processor(AbstractProcessor):
	def __init__(self, params=params['process'], verbose=0):
		AbstractProcessor.__init__(self, verbose=verbose)
		self.params = params

	def process(self, data, labels, name):
		eeg, neck, mass = data
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
		ds.data -= np.mean(ds.data, axis=0)
		if labels is not None:
			if self.verbose > 1 : print("Adding Labels")
			labels = Dataset(label_names=['LABELS'], labels=labels.data[:-1].reshape(-1,1))
			ds = ds.concatenate(labels)
		ds.name = name
		ds.clean()
		return ds

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
