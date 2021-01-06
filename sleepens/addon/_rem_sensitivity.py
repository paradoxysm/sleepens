import numpy as np

from sleepens.addon._base_addon import AbstractAddon

class RemFix(AbstractAddon):
	def __init__(self, verbose=0):
		AbstractAddon.__init__(self, verbose=verbose)

	def addon(self, Y_hat, p, avg_threshold=0.08, init_threshold=0.03, min_threshold=0.02,
					window=2, min_size=3):
		rem_count = 0
		i = len(p) - window
		while i >= 0:
			if p[i] == 3 : rem_count += 1
			if rem_count == 0 and p[i+1] == 2 and Y_hat[i,3] >= init_threshold:
				avg = np.mean(Y_hat[i:i+window,3])
				if avg >= avg_threshold : rem_count += 1
				else : rem_count = 0
			if rem_count > 0 and p[i] != 0 and p[i] != 3 and Y_hat[i,3] >= min_threshold:
				avg = np.mean(Y_hat[i:i+window,3])
				if avg >= avg_threshold:
					rem_count += 1
					if rem_count > min_size : p[i] = 3
				else : rem_count = 0
			elif p[i] != 3 : rem_count = 0
			if rem_count == min_size : p[i:i+min_size] = 3
			i -= 1
		return p

'''
Go backwards
If has been above 0.5? or the last one is above 0.5? = threshold is 0.3 to extend
If not = threshold is 0.075? or 0.1? or 0.06? or 0.03? for at least 3? or 2? epochs to create
Cannot extend a REM by more than 3? no limit?

or some averaging threshold = average must be above 0.08? 0.07? 0.06? etc? across at least 4? 3? 2? epochs
ROlling average size 3? or 2?
Starting threshold of 0.03 then begin averaging
Keep going so long as average meets threshold and current is above 0.02?
Don't replace AW? Don't replace QW if its later in forward direction? (can replace QW but only after replacing NR)

Also implement QW+AW sensitivity?

Order:
1. REM sensitivity
2. QW+AW sensitivity
3. Transition
'''
