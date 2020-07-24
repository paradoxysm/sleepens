"""Signal Processing Functions"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from scipy import signal

from sleepens.utils.data.normalize import prob_normalize

def compute_bands(f, Pxx, bands, compute='sum'):
	"""
	Compute the power spectrum within
	specified banding intervals.

	Parameters
	----------
	f : ndarray
		Array of frequencies.

	Pxx : ndarray
		Power spectral density of `data`.

	bands : dict
		Dictionary of band intervals to evaluate.

	compute : {'mean', 'median', 'max', 'min', 'sum'}, default='mean'
		Method to compute the summary statistic of the band.

	Returns
	-------
	dataBands : dict
		A dictionary of computed power values at each band,
		matching the bands provided in `bands`.
	"""
	data_bands = {}
	for band in bands.keys():
		f_ix = np.where((f >= bands[band][0]) & (f <= bands[band][1]))[0]
		if compute == 'mean':
			data_bands[band] = np.mean(Pxx[f_ix])
		elif compute == 'median':
			data_bands[band] = np.median(Pxx[f_ix])
		elif compute == 'max':
			data_bands[band] = np.max(Pxx[f_ix])
		elif compute == 'min':
			data_bands[band] = np.min(Pxx[f_ix])
		elif compute == 'sum':
			data_bands[band] = np.sum(Pxx[f_ix])
		else:
			raise ValueError("Provided compute argument must be 'mean','median','max','min', or 'sum'")
	return data_bands

def welch(data, fs=1.0, nperseg=None, noverlap=None, nfft=None, detrend='constant'):
	"""
	Estimate the power spectral density via Welch's method
	and Hann windows.

	Welch’s method computes an estimate of the power spectral
	density by dividing the data into overlapping segments,
	computing a modified periodogram for each segment and
	averaging the periodograms.

	Parameters
	----------
	data : array-like
		Time series of data values.

	fs : float, default=1.0
		Sampling frequency of `data`.

	nperseg : int or None, default=None
		Length of each segment.

	noverlap : int or None, default=None
		Number of samples to overlap between segments.
		If None, `noverlap` is half of `nperseg`.

	nfft : int or None, default=None
		Length of the FFT to be used. If None, it is
		set to `nperseg`.

	detrend : str or function or False, default='constant'
		Specifies detrend method for the data.
		If detrend is a string, it is passed as the type
		argument to the detrend function. If it is a function,
		it takes a segment and returns a detrended segment.
		If detrend is False, no detrending is done.

	Returns
	-------
	f : ndarray, shape=(n_frequencies,)
		Array of frequencies.

	Pxx : ndarray, shape=(n_frequencies,)
		Power spectral density of `data`.
	"""
	f, Pxx = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap,
							nfft=nfft, detrend=detrend)
	return f, Pxx

def spectrogram(data, fs=1.0, nperseg=None, noverlap=None, nfft=None, detrend='constant'):
	"""
	Compute a spectrogram of consecutive Fourier Transforms,
	using a Turkey window of shape 0.25

	Parameters
	----------
	data : array-like
		Time series of data values.

	fs : float, default=1.0
		Sampling frequency of `data`.

	nperseg : int or None, default=None
		Length of each segment.

	noverlap : int or None, default=None
		Number of samples to overlap between segments.
		If None, `noverlap` is half of `nperseg`.

	nfft : int or None, default=None
		Length of the FFT to be used. If None, it is
		set to `nperseg`.

	detrend : str or function or False, default='constant'
		Specifies detrend method for the data.
		If detrend is a string, it is passed as the type
		argument to the detrend function. If it is a function,
		it takes a segment and returns a detrended segment.
		If detrend is False, no detrending is done.

	Returns
	-------
	f : ndarray, shape=(n_frequencies,)
		Array of frequencies.

	t : ndarray, shape=(n_segments,)
		Array of segment times.

	Sxx : ndarray, shape=(n_segments, n_frequencies)
		Spectrogram of `data`.
	"""
	f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap,
							nfft=nfft, detrend=detrend)
	return f, t, Sxx.T

def spectral_entropy(f, Pxx):
	"""
	Compute the spectral entropy of the power spectral density
	according to Shannon entropy.

	Parameters
	----------
	f : ndarray
		Array of frequencies.

	Pxx : ndarray
		Power spectral density.

	Returns
	-------
	entropy : float
		Spectral entropy between 0 and 1, with 0
		representing no entropy and 1 as maximal entropy.
	"""
	norm_Pxx = prob_normalize(Pxx)
	entropy = np.sum(norm_Pxx*np.log(1/norm_Pxx))
	normEntropy = entropy / np.log(len(f))
	return normEntropy

def rms(data):
	"""
	Calculate the root mean square of the data.

	Parameters
	----------
	data : array-like
		Time series of data values.

	Returns
	-------
	rms : float
		root mean square of the given data.
	"""
	return np.sqrt(np.mean(np.square(data)))

def zero_cross(data, zero=0):
	"""
	Calculate the number of times the data
	crosses the `zero` value.

	Parameters
	----------
	data : ndarray
		Time series of data values.

	zero : scalar, default=0
		Crossing point to analyze.

	Returns
	-------
	cross : float
		Number of times the data crosses `zero`.
	"""
	return (((data[:-1] - zero) * (data[1:] - zero)) < zero).sum()

def percentile_mean(data, k):
	"""
	Compute the mean of data above the k-th percentile.

	Parameters
	----------
	data : ndarray
		Time series of data values.

	k : float
		Percentile threshold.

	Returns
	-------
	mean : float
		Mean of the data above the k-th percentile.
	"""
	index = (len(data)*k)//100
	return np.mean(np.partition(data, index)[index:])
