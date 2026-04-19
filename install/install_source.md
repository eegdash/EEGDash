<a id="install-source"></a>

# Installing from sources

If you want to test features under development or contribute to the library, or if you want to test the new tools that have been tested in EEGDash and not released yet, this is the right tutorial for you!

For an overview of contributor workflows and project internals, see [Developer Notes](../developer_notes.md).

#### NOTE
If you are only trying to install EEGDash, we recommend the [Installing from PyPI](install_pip.md) section for details on that.

## Install preview version from PyPI

```shell
pip install --pre eegdash
```

You should will install the version of eegdash that is currently under development at main branch, which may not be stable.

## Install directly from repository from GitHub

Let’s suppose that you want to install EEGDash from the source. The first thing you should do is clone the EEGDash repository to your computer and enter inside the repository.

```shell
git clone https://github.com/eegdash/EEGDash && cd EEGDash
```

You should now be in the root directory of the EEGDash repository.

## Installing EEGDash from the source with pip

If you want to only install EEGDash from source once and not do any development
work, then the recommended way to build and install is to use `pip`

For the latest development version, directly from GitHub:

```shell
pip install git+https://github.com/eegdash/EEGDash.git
```

If you have a local clone of the EEGDash git repository:

```shell
pip install -e .
```

This will install EEGDash in editable mode, i.e., changes to the source code could be used
directly in python.

You could also install optional dependency, like to import datasets from test and docs.

```shell
pip install -e .[test,docs,dev]
```

There is also optional dependencies for unit testing and building documentation, you could install
them if you want to contribute to EEGDash.

```shell
pip install -e .[all]
```

# Testing if your installation is working

To verify that EEGDash is installed and running correctly, run the following command:

```shell
python -m "import eegdash; eegdash.__version__"
```

<!-- This (-*- rst -*-) format file contains commonly used link targets
and name substitutions.  It may be included in many files,
therefore it should only contain link targets and name
substitutions.  Try grepping for "^\.\. _" to find plausible
candidates for this list. -->
<!-- NOTE: reST targets are
__not_case_sensitive__, so only one target definition is needed for
nipy, NIPY, Nipy, etc... -->
<!-- braindecode -->
<!-- mne -->
<!-- moabb -->
<!-- main dependencies -->
<!-- git stuff -->
<!-- other stuff -->
<!-- vim: ft=rst -->
