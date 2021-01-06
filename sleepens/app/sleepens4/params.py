from sleepens.utils.data import normalize as n

params = {
	'reader': {
		'EEG_NAME': 'EEG',
		'NECK_NAME': 'NECK',
		'MASS_NAME': 'MASS',
		'SCORE_NAME': 'SCORE',
		'SCORE_MAP': { 'AW': 0, 'QW': 1, 'NR': 2, 'R': 3 },
	},
	'process': {
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
}
