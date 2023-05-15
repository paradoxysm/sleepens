## Sleep Ensemble

[![CircleCI](https://img.shields.io/circleci/build/github/paradoxysm/sleepens?style=flat-square)](https://circleci.com/gh/paradoxysm/sleepens/tree/master)
[![Codecov](https://flat.badgen.net/codecov/c/github/paradoxysm/sleepens?label=coverage&kill_cache=1)](https://codecov.io/gh/paradoxysm/sleepens)
[![DOI](https://img.shields.io/badge/DOI-10.5821%2Fzenodo.7791521-blue?style=flat-square)](https://zenodo.org/badge/latestdoi/282098794)
[![GitHub](https://img.shields.io/github/license/paradoxysm/sleepens?color=blue&style=flat-square)](https://github.com/paradoxysm/sleepens/blob/master/LICENSE)

Sleep Ensemble is a framework for end-to-end sleep state classification using machine learning. It is designed to allow for modular data processing, classification, and further post-processing.
Published in [SLEEP](https://academic.oup.com/sleep/advance-article-abstract/doi/10.1093/sleep/zsad101/7109541).

## Installation

Install a suitable python environment from [python.org](https://www.python.org/downloads/release/python-378/).
> Sleep Ensemble supports Python 3.7 only. It is extensively tested and developed with 64-bit Python 3.7.8 on Windows.

> Sleep Ensemble pre-trained builds are only useable for the specific OS and 32/64-bit Python environment. The included pre-trained SleepEnsemble4 is built on 64-bit Python 3.7.8 on Windows.

Install the latest `sleepens` release using `pip` (on a terminal like command prompt):
```
pip install sleepens
```
Alternatively, download the source code for the latest release [here](https://github.com/paradoxysm/sleepens/releases). Unzip into desired location. Using the terminal, navigate to the top `sleepens` folder where `setup.py` is located and run the following:
```
python setup.py install
```

Download any pre-trained builds [here](https://github.com/paradoxysm/sleepens/blob/master/BUILDS.md) or train your own.

## Usage

Sleep Ensemble is built for easy use with a text-based python script that you can run in a command terminal. This application allows you to quickly classify data or train new models.

The application runs via `sleepens.py` which can be copied/moved anywhere as needed. To run, using the terminal, navigate to `sleepens.py` and run:
```
python sleepens.py
```

As a framework, the Sleep Ensemble package can be used as you would any other package. The end-user program can be accessed:
```python
import sleepens
sleepens.run()
```

Alongside the Sleep Ensemble application, the framework is designed for high modularity and integration with other scripts in very little code.

```python
# Setup a protocol
from sleepens.protocols.sleepens4 import SleepEnsemble4
model = SleepEnsemble4()

# Load a pre-trained model in a .joblib file
import joblib
model.classifier = joblib.load("/path/to/model.joblib")

# Classify your data
# Data is a list of 2D arrays in the form of (n samples, n features)
predictions = model.predict(data)
```

For full details on usage, see the [documentation](https://github.com/paradoxysm/sleepens/tree/master/doc).

## Changelog

See the [changelog](https://github.com/paradoxysm/sleepens/blob/master/CHANGES.md) for a history of notable changes to `sleepens`.

## Development

[![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/paradoxysm/sleepens?style=flat-square&kill_cache=1)](https://codeclimate.com/github/paradoxysm/sleepens/maintainability)

`sleepens` is in a relatively finished state. It has not been tested on different Python environment and OS combinations.

Currently, `sleepens` supports .mat, .smr/.smrx, .xls file formats for reading and writing. Of note, presently the SONpy interface for reading/writing .smr/.smrx files works for reading .smr only and writing .smrx only. No clear documentation exists to explain errors when attempting to read from .smrx. Additional i/o interfaces can be contributed to `sleepens.io.interfaces` following the basic structure.

The underlying data processing pipelines and classification models are modular and can be adjusted to create different sleep ensemble models. `sleepens` currently contains one protocol, SleepEnsemble4, for 4-state sleep classification. New or modified protocols can be contributed to `sleepens.protocols` following the basic structure.

Finally, `sleepens` uses `joblib` to store the pre-trained builds. This isn't the most secure method nor is the most data storage efficient. Ideally, a custom parameter export/load method is implemented that can reinstate a pre-trained build.

## Dependencies

`sleepens` was developed using Python 3.7.8. Since development, `scikit-learn` has updated to a point that `sleepens` is not compatible with later versions of `scikit-learn` due to attribute name changes. At the same time, `sonpy` is limited to Python 3.7 to 3.9. For the sake of replicability (especially with pre-trained models provided on publication, the repository remains using the below dependencies at these versions. This also limits Python to 3.7.

```
numpy==1.21.6
scikit-learn==0.24.0
joblib==1.0.0
tqdm==4.55.0
xlrd==2.0.1
xlwt==1.3.0
sonpy==1.7.5
scipy==1.7.2
```

Should one wish to upgrade these dependencies to the latest version of `scikit-learn`, change all references of `n_features_` to `n_features_in_` in the following:
- `sleepens/sleepens/ml/_base_model.py`
- `sleepens/sleepens/ml/models/_gb.py`
- `sleepens/sleepens/ml/models/_tsens.py`
- `sleepens/sleepens/ml/models/_stsens.py`

Furthermore, change the loss parameter in `sleepens/sleepens/ml/_gb.py` line 220 to 'log_loss' (which replaces 'deviance' but is algorithmically the same). 

## Help and Support

### Documentation

Documentation for `sleepens` can be found [here](https://github.com/paradoxysm/sleepens/tree/master/doc).

### Issues and Questions

Issues and Questions should be posed to the issue tracker [here](https://github.com/paradoxysm/sleepens/issues).
