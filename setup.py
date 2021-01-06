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
	version='0.1.0',
    description='Sleep Classification using Ensemble Classification',
    long_description=(read('README.md') + '\n\n'),
	long_description_content_type="text/markdown",
	url='http://github.com/paradoxysm/sleepens',
	download_url = 'https://github.com/paradoxysm/sleepens/archive/0.1.0.tar.gz',
    author='paradoxysm',
	author_email='paradoxysm.dev@gmail.com',
    license='BSD-3-Clause',
    packages=find_packages(),
    install_requires=[
		'numpy',
		'tqdm',
		'sklearn',
        'xlrd',
        'xlwt',
		'joblib'
    ],
	python_requires='>=3.4, <4',
	classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
		'Topic :: Scientific/Engineering :: Information Analysis',
		'Intended Audience :: Science/Research',
    ],
	keywords=['python', 'ml', 'ensemble', 'sleep', 'classification'],
    zip_safe=True)
