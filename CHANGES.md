# Changelog

### Legend

- ![Feature](https://img.shields.io/badge/-Feature-blueviolet?style=flat-square) : Something that you couldn’t do before.
- ![Enhancement](https://img.shields.io/badge/-Enhancement-purple?style=flat-square) : A miscellaneous minor improvement.
- ![Efficiency](https://img.shields.io/badge/-Efficiency-indigo?style=flat-square) : An existing feature now may not require as much computation or memory.
- ![Fix](https://img.shields.io/badge/-Fix-red?style=flat-square) : Something that previously didn’t work as documented or as expected should now work.
- ![Documentation](https://img.shields.io/badge/-Documentation-blue?style=flat-square) : An update to the documentation.
- ![Other](https://img.shields.io/badge/-Other-lightgrey?style=flat-square) : Miscellaneous updates such as package structure or GitHub quality of life updates.

### Version 1.0.4
- ![Fix](https://img.shields.io/badge/-Fix-red?style=flat-square) : Froze all dependency versions; included `threadpoolctl` version requirement.
- ![Other](https://img.shields.io/badge/-Other-lightgrey?style=flat-square) : Fixed dependency naming and requirements in `setup.py` and `requirements.txt`.
- ![Other](https://img.shields.io/badge/-Other-lightgrey?style=flat-square) : Updated model naming schema when loading a model to use the file name as the model name.
- ![Documentation](https://img.shields.io/badge/-Documentation-blue?style=flat-square) : Updated relevant documentation to incorporate changes.


### Version 1.0.3
- ![Documentation](https://img.shields.io/badge/-Documentation-blue?style=flat-square) : Updated README to reflect freezing of dependencies.
- ![Other](https://img.shields.io/badge/-Other-lightgrey?style=flat-square) : Specified exact versions of dependencies in `requirements.txt` and in `setup.py`.

### Version 1.0.1
- ![Fix](https://img.shields.io/badge/-Fix-red?style=flat-square) : Remove `min_impurity_split` from `GradientBoostingClassifier` as it no longer exists in the underlying `sklearn` class.
- ![Documentation](https://img.shields.io/badge/-Documentation-blue?style=flat-square) : Improve organization of the build directory.

### Version 1.0.0
- ![Feature](https://img.shields.io/badge/-Feature-blueviolet?style=flat-square) : Sleep Ensemble is released! Hurray!
- ![Feature](https://img.shields.io/badge/-Feature-blueviolet?style=flat-square) : IO interface selection for read and write implemented. Interfaces for .mat exports from Spike2, .smr/.smrx from Spike2, and .xls "datasets" have been included.
- ![Feature](https://img.shields.io/badge/-Feature-blueviolet?style=flat-square) : End-user command line interface application developed. Includes ability to classify data, train/validate models, load/export builds.
- ![Feature](https://img.shields.io/badge/-Feature-blueviolet?style=flat-square) : Protocol management embedded into CLI application similar to IO interface integration.
- ![Feature](https://img.shields.io/badge/-Feature-blueviolet?style=flat-square) : Sleep Ensemble 4 protocol added to [`sleepens.protocols`](https://github.com/paradoxysm/sleepens/tree/master/sleepens/protocols).
- ![Documentation](https://img.shields.io/badge/-Documentation-blue?style=flat-square) : [`README.md`](https://github.com/paradoxysm/sleepens/bloc/master/README.md) includes more details on installation and usage.
- ![Documentation](https://img.shields.io/badge/-Documentation-blue?style=flat-square) : [`BUILDS.md`](https://github.com/paradoxysm/sleepens/bloc/master/BUILDS.md) written as a build directory to easily find pre-trained build binaries from release history.
