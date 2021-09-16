## Sleep Ensemble

[![Travis](https://flat.badgen.net/travis/paradoxysm/sleepens?label=build&kill_cache=1)](https://app.travis-ci.com/github/paradoxysm/sleepens)
[![Codecov](https://flat.badgen.net/codecov/c/github/paradoxysm/sleepens?label=coverage&kill_cache=1)](https://codecov.io/gh/paradoxysm/sleepens)
[![GitHub](https://flat.badgen.net/github/license/paradoxysm/sleepens)](https://github.com/paradoxysm/sleepens/blob/master/LICENSE)

Sleep Ensemble is a framework for end-to-end sleep state classification using machine learning. It is designed to allow for modular data processing, classification, and further post-processing.

Sleep Ensemble is built upon a novel ensemble architecture, the Time Series Ensemble, which provides greater performance and generalizability on sleep time series data over other models (e.g. gradient boosting, random forests, neural networks, etc.).

## Installation

Install a suitable python environment from [python.org](https://www.python.org/downloads/release/python-378/).
> Sleep Ensemble supports Python 3.7 or later.
> Sleep Ensemble is extensively tested and developed in Python 3.7.8 on Windows 10. The included pre-trained SleepEnsemble4 build is thus only useable on a 64bit Python version (specifically Python 3.7.x) on Windows. Its use may be possible with other Python 3.x versions but not guaranteed. You can train your own build on any of the supported Python environments in any OS.

Download the latest `sleepens` release [here](https://github.com/paradoxysm/sleepens/releases). Unzip into desired location.

Using a terminal like command prompt, navigate to the top `sleepens` folder where `setup.py` is located.
Run the following:
```
python setup.py install
```
> Your installation of Python may require you to use the alias python3 to run python scripts.

## Usage

Sleep Ensemble is built for easy use with a text-based python script that you can run in a command terminal. This application allows you to quickly classify data or train new models.

The application runs via `sleepens.py` which can be copied/moved anywhere as needed. To run, using the terminal, navigate to `sleepens.py` and run:
```
python sleepens.py
```

Alongside the Sleep Ensemble application, the framework is designed for high modularity and integration with other scripts in very little code.

```python
# Load a pre-trained model in a .joblib file
import joblib
model = joblib.load("/path/to/model.joblib")

# Classify your data
# Data is a list of 2D arrays in the form of (n samples, n features)
predictions = model.predict(data)
```

The underlying data processing pipelines and classification models are modular and can be adjusted to create different sleep ensemble models. These pipelines can be contributed to `sleepens.protocols`

For full details on usage, see the [documentation](https://github.com/paradoxysm/sleepens/tree/master/doc).

### For Development

Since the release package is the source code, you can also develop the package in this manner. You may also install `sleepens` using `pip`:
```
pip install sleepens
```
As a framework, the Sleep Ensemble package can be used as you would any other package. The end-user program can be accessed:
```python
import sleepens
sleepens.run()
```

## Changelog

See the [changelog](https://github.com/paradoxysm/sleepens/blob/master/CHANGES.md) for a history of notable changes to `sleepens`.

## Development

[![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/paradoxysm/sleepens?style=flat-square&kill_cache=1)](https://codeclimate.com/github/paradoxysm/sleepens/maintainability)

`sleepens` is in a relatively finished state. It has not been tested on different Python environment and OS combinations.

## Help and Support

### Documentation

Documentation for `sleepens` can be found [here](https://github.com/paradoxysm/sleepens/tree/master/doc).

### Issues and Questions

Issues and Questions should be posed to the issue tracker [here](https://github.com/paradoxysm/sleepens/issues).
