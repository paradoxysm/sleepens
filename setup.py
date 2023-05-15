import os
from setuptools import setup, find_packages

def read(*paths):
	"""
	Build a file path from *paths* and return the contents.
	"""
	with open(os.path.join(*paths), 'r') as f:
		return f.read()

setup(
	name='sleepens',
	version='1.0.2',
	description='Sleep Classification using Ensemble Classification',
	long_description=(read('README.md') + '\n\n'),
	long_description_content_type="text/markdown",
	url='http://github.com/paradoxysm/sleepens',
	download_url = 'https://github.com/paradoxysm/sleepens/archive/1.0.2.tar.gz',
	author='paradoxysm',
	author_email='paradoxysm.dev@gmail.com',
	license='BSD-3-Clause',
	packages=find_packages(exclude=['tests']),
	include_package_data=True,
	install_requires=[
		'numpy==1.21.6',
		'scikit-learn==0.24.0',
		'joblib==1.1.1',
		'tqd==4.55.0',
		'xlrd==2.0.1',
		'xlwt==1.3.0',
		'sonpy==1.7.5',
		'scipy==1.7.2',
	],
	python_requires='>=3.9, <3.10',
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Developers',
		'Natural Language :: English',
		'License :: OSI Approved :: BSD License',
		'Operating System :: OS Independent',
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
		'Topic :: Software Development :: Libraries :: Python Modules',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
		'Topic :: Scientific/Engineering :: Information Analysis',
		'Intended Audience :: Science/Research',
	],
	keywords=['python', 'ml', 'ensemble', 'sleep', 'classification'],
	zip_safe=True)
