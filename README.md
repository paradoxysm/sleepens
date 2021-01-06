## Sleep Ensemble

[![Travis](https://flat.badgen.net/travis/paradoxysm/sleepens?label=build)](https://travis-ci.com/paradoxysm/sleepens)
[![Codecov](https://flat.badgen.net/codecov/c/github/paradoxysm/sleepens?label=coverage)](https://codecov.io/gh/paradoxysm/sleepens)
[![GitHub](https://flat.badgen.net/github/license/paradoxysm/sleepens)](https://github.com/paradoxysm/sleepens/blob/master/LICENSE)

Sleep Ensemble is a framework for flexible sleep state classification using ensemble learning. It is designed to allow for modular data processing, classification, and further post-processing.

Sleep Ensemble is built upon a novel ensemble architecture, the Time Series Ensemble, which provides greater performance and generalizability on sleep time series data over other models (e.g. gradient boosting, random forests, neural networks, etc.).

## Installation

Once you have a suitable python environment setup, `sleepens` can be easily installed using `pip`:
```
pip install sleepens
```
> `sleepens` is tested and supported on Python 3.5 up to Python 3.8. Usage on other versions of Python is not guaranteed to work as intended.

Then download the latest application package from [releases](https://github.com/paradoxysm/sleepens/releases) and run via `python sleepens.py`. The package contains the latest Sleep Ensemble models to use.

## Usage

Sleep Ensemble is built for easy use with a text-based python script that you can run in a command terminal. This application allows you to quickly classify data or train new models.

Alongside the Sleep Ensemble application, the framework is designed for high modularity and integration with other scripts in very little code.

```python
# Import the base shell sleep ensemble and load a pre-trained model in a .joblib file
from sleepens import SleepEnsemble
model = SleepEnsemble().load("/path/to/model.joblib")

# Classify your data
predictions = model.predict(data)
```

The underlying data processing pipelines and classification models are modular and can be adjusted to create different sleep ensemble models.

For full details on usage, see the [documentation](https://github.com/paradoxysm/metanetwork/tree/master/doc).

## Changelog

See the [changelog](https://github.com/paradoxysm/sleepens/blob/master/CHANGES.md) for a history of notable changes to `sleepens`.

## Development

[![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/paradoxysm/sleepens?style=flat-square)](https://codeclimate.com/github/paradoxysm/sleepens/maintainability)

`sleepens` is in heavy development. Don't look, it's embarrassing!

## Help and Support

### Documentation

Documentation for `sleepens` can be found [here](https://github.com/paradoxysm/sleepens/tree/master/doc).

### Issues and Questions

Issues and Questions should be posed to the issue tracker [here](https://github.com/paradoxysm/sleepens/issues).
