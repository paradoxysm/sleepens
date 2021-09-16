import numpy as np

from sleepens.process import primary as prim
from sleepens.process import secondary as sec
from sleepens.utils.data import sampling as s
from sleepens.utils.data import normalize as n
from sleepens.io import Dataset

def process(data, labels, name, params, verbose=0):
	eeg, neck, mass = data
	if verbose > 0 : print("Extracting Features")
	if verbose > 1 : print("Detrending Data")
	eeg, neck, mass = _detrend(eeg, neck, mass)
	if verbose > 1 : print("Conducting FFT Analysis")
	eeg_fft = _fft(params['EEG_FFT'], params['EPOCH_SIZE'], eeg)[0]
	neck_fft, mass_fft = _fft(params['EMG_FFT'], params['EPOCH_SIZE'], neck, mass)
	if verbose > 1 : print("Calculating RMS")
	eeg_rms, neck_rms, mass_rms = _rms(params['EPOCH_SIZE'], eeg, neck, mass)
	if verbose > 1 : print("Calculating Percentile Mean")
	neck_prct, mass_prct = _prct(params['EPOCH_SIZE'], params['PRCTILE'], neck, mass)
	if verbose > 1 : print("Calculating Epoched Variance")
	neck_twitch, mass_twitch = _ep_var(params['EPOCHED_VAR'], params['EPOCH_SIZE'], neck, mass)
	if verbose > 1 : print("Calculating Spectral Entropy")
	eeg_entropy, neck_entropy, mass_entropy = _entropy(eeg_fft, neck_fft, mass_fft)
	if verbose > 1 : print("Calculating Frequency Bands")
	eeg_bands = sec.spectral_band(eeg_fft, bands=params['BANDS'], merge=params['BAND_MERGE'])
	if verbose > 1 : print("Calculating Frequency Ratios")
	ratios = sec.ratio(eeg_bands, ratios=params['RATIOS'])
	if verbose > 1 : print("Merging")
	emg_rms = _merge(mass_rms, neck_rms, feature='EMG RMS',
								method=params['EMG_RMS_MERGE'])
	emg_prct = _merge(mass_prct, neck_prct, feature='EMG PRCT',
								method=params['EMG_PRCT_MERGE'])
	emg_twitch = _merge(mass_twitch, neck_twitch, feature='EMG TWITCH',
								method=params['EMG_TWITCH_MERGE'])
	emg_entropy = _merge(mass_entropy, neck_entropy, feature='EMG ENTROPY',
								method=params['EMG_ENTROPY_MERGE'])
	features = [eeg_bands, ratios, eeg_rms, emg_rms, emg_prct,
					eeg_entropy, emg_entropy, emg_twitch]
	if verbose > 1 : print("Creating Dataset")
	ds = _create_ds(features, overlap=False)
	if verbose > 1 : print("Transforming Data")
	ds = sec.transform(ds, ops=params['TRANSFORM'])
	ds.data -= np.mean(ds.data, axis=0)
	if labels is not None:
		if verbose > 1 : print("Adding Labels")
		labels = Dataset(label_names=['LABELS'], labels=labels.data[:-1].reshape(-1,1))
		ds = ds.concatenate(labels)
	ds.name = name
	ds.clean()
	return ds

def _detrend(*channels):
	for c in channels:
		c.process(n.demean)
	return channels

def _fft(params, epoch_size, *channels):
	result = []
	for c in channels:
		fft = prim.fft(c, epoch_size,
						nperseg_factor=params['NPERSEG'],
						noverlap_factor=params['NOVERLAP'],
						detrend=params['DETREND'])
		result.append(fft)
	return result

def _rms(epoch_size, *channels):
	result = []
	for c in channels:
		rms = prim.rms(c, epoch_size)
		result.append(rms)
	return result

def _prct(epoch_size, k, *channels):
	result = []
	for c in channels:
		prct = prim.percentile_mean(c, epoch_size,
						k=k)
		result.append(prct)
	return result

def _ep_var(params, epoch_size, *channels):
	sub_epoch_size = params['SUBEPOCH_SIZE']
	threshold = params['THRESHOLD']
	merge = params['MERGE']
	result = []
	for c in channels:
		epvar = prim.epoched_variance(c, epoch_size,
						sub_epoch_size= sub_epoch_size,
						threshold=threshold, merge=merge)
		result.append(epvar)
	return result

def _entropy(*channels):
	result = []
	for c in channels:
		entropy = sec.spectral_entropy(c)
		result.append(entropy)
	return result

def _merge(channel1, channel2, feature='MERGE', method='mean'):
	if method is None:
		raise ValueError("Feature")
	merge = channel1.concatenate(channel2, overlap=False)
	return sec.merge(merge, feature=feature, method=method)

def _create_ds(features, overlap=False):
	if len(features) == 0 : return Dataset()
	ds = features.pop(0)
	for f in features:
		ds = ds.concatenate(f, overlap=overlap)
	return ds
