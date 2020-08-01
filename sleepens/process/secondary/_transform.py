"""Transform Processor"""

# Authors: Jeffrey Wang
# License: BSD 3 clause

from sleepens.io import Dataset

def transform(ds, name=None, ops=[]):
	"""
	Transform feature data according
	to parameters. Transformation
	involves normalization, smoothing,
	and/or transformation.

	Parameters
	----------
	data : ndarray, shape=(n_epochs,)
		Feature data.

	ops : list
		Ordered list of transformation operations
		to perform on `data`. Each operation
		is represented by a dictionary of the function
		followed by applicable arguments.

	Returns
	-------
	transformed : ndarray, shape=(n_epochs,)
		Data after transformation.
	"""
	data = ds.data
	for op in ops:
		func, args, kwargs = None, [], {}
		if 'FUNCTION' in op.keys() : func = op['FUNCTION']
		if 'ARGS' in op.keys() : args = op['ARGS']
		if 'KWARGS' in op.keys() : kwargs = op['KWARGS']
		if func is not None : data = func(data, *args, **kwargs)
	return Dataset(name=name, features=ds.features, data=data)
