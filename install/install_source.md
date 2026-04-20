<a id="install-source"></a>

# Installing from sources

This page covers installing EEGDash from source, which is what you want for contributing or for trying features that have not yet been released.

For an overview of contributor workflows and project internals, see [Developer Notes](../developer_notes.md).

#### NOTE
If you only want to install a released version, see [Installing from PyPI](install_pip.md).

## Install a pre-release from PyPI

```shell
pip install --pre eegdash
```

This installs the in-development version of `eegdash` from the main branch. It may not be stable.

## Install directly from GitHub

Clone the repository and change into it:

```shell
git clone https://github.com/eegdash/EEGDash && cd EEGDash
```

## Install with pip

For a one-off install straight from GitHub:

```shell
pip install git+https://github.com/eegdash/EEGDash.git
```

From a local clone, install in editable mode so source edits are picked up without reinstalling:

```shell
pip install -e .
```

Optional extras let you pull in test and documentation dependencies:

```shell
pip install -e .[test,docs,dev]
```

Or install everything, which is what you want for contributing:

```shell
pip install -e .[all]
```

## Verifying the installation

```shell
python -c "import eegdash; print(eegdash.__version__)"
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
