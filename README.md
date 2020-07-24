## Sleep Ensemble

[![Travis](https://flat.badgen.net/travis/paradoxysm/sleepens?label=build)](https://travis-ci.com/paradoxysm/sleepens)
[![Codecov](https://flat.badgen.net/codecov/c/github/paradoxysm/sleepens?label=coverage)](https://codecov.io/gh/paradoxysm/sleepens)
[![GitHub](https://flat.badgen.net/github/license/paradoxysm/sleepens)](https://github.com/paradoxysm/sleepens/blob/master/LICENSE)

Sleep Ensemble is a framework for flexible sleep state classification using powerful ensemble learning.

## Installation

Once you have a suitable python environment setup, `sleepens` can be easily installed using `pip`:
```
pip install sleepens
```
> `sleepens` is tested and supported on Python 3.4 up to Python 3.8. Usage on other versions of Python is not guaranteed to work as intended.

## Usage

Sleep Ensemble is designed for easy use with as little as a single line of code.

```python
# Import the pre-trained Sleep Ensemble model
from sleepens import SleepEnsemble

# Classify your data
predictions = SleepEnsemble.predict(data)
```

Sleep Ensemble contains three versions:
1. A pre-trained model
2. An untrained version ready to calibrate
3. An online model for continuous learning.
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
