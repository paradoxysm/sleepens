import numpy as np
from abc import ABC, abstractmethod

class Base:
	"""
	Base Class
	"""
	def __init__(self):
		pass

	def get_params(self, seed=False):
		"""
		Get all parameters of the object, recursively.

		Parameters
		----------
		seed : bool, default=False
			Determines if the current state of
			any stored RandomStates should be preserved.

		Returns
		-------
		params : dict
			Dictionary of object parameters.
		"""
		params = vars(self)
		for k in params.keys():
			if hasattr(params[k], 'get_params'):
				params[k] = dict(list(params[k].get_params().items()) + \
								[('type', type(params[k]))])
			elif isinstance(params[k], np.random.RandomState):
				seed = params[k].get_state() if seed else None
				params[k] = {'type': np.random.RandomState,
								'seed': seed}
			elif hasattr(params[k], '__dict__'):
				params[k] = vars_recurse(params[k])
		params = dict(list(params.items()) + [('type', type(self))])
		return params

	def set_params(self, params):
		"""
		Set the attributes of the object with the given
		parameters.

		Parameters
		----------
		params : dict
			Dictionary of object parameters.

		Returns
		-------
		self : Base
			Itself, with parameters set.
		"""
		valid = self.get_params().keys()
		for k, v in params.items():
			if k not in valid:
				raise ValueError("Invalid parameter %s for object %s" % \
									(k, self.__name__))
			param = v
			if isinstance(v, dict) and 'type' in v.keys():
				t = v['type']
				if t == np.random.RandomState:
					state = v['seed']
					if state is None : param = np.random.RandomState()
					else : param = np.random.RandomState().set_state(state)
				elif 'set_params' in dir(t):
					v.pop('type')
					param = t().set_params(v)
				else:
					param = t()
					for p, p_v in v.pop('type').items():
						setattr(param, p, p_v)
			print(k, param)
			setattr(self, k, param)
		return self

def vars_recurse(obj):
	"""
	Recursively collect vars() of the object.

	Parameters
	----------
	obj : object
		Object to collect attributes

	Returns
	-------
	params : dict
		Dictionary of object parameters.
	"""
	if hasattr(obj, '__dict__'):
		params = vars(obj)
		for k in params.keys():
			if hasattr(params[k], '__dict__'):
				params[k] = vars_recurse(params[k])
		params = dict(list(params.items()) + [('type', type(obj))])
		return params
	raise ValueError("obj does not have __dict__ attribute")
