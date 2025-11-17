# Contributing to EEG-Dash

Thank you for your interest in contributing to EEG-Dash! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Commit Message Conventions](#commit-message-conventions)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project is a collaborative initiative between UCSD and Ben-Gurion University, supported by the NSF. We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Familiarity with EEG data and BIDS format (helpful but not required)

### Finding Issues to Work On

- Check the [issue tracker](https://github.com/sccn/EEG-Dash-Data/issues) for open issues
- Look for issues labeled `good first issue` or `help wanted`
- Before starting work on a new feature, open an issue to discuss it

## Development Setup

### 1. Fork and Clone the Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/EEG-Dash-Data.git
cd EEG-Dash-Data

# Add the upstream repository
git remote add upstream https://github.com/sccn/EEG-Dash-Data.git
```

### 2. Create a Development Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Unix/macOS)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate

# Install the package in editable mode with all development dependencies
pip install -e .[all]
```

### 3. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install
```

This will automatically run code quality checks (ruff, codespell) before each commit.

### 4. Verify Installation

```bash
# Run tests to ensure everything is set up correctly
pytest tests/ -v

# Check that you can import eegdash
python -c "from eegdash import EEGDash; print('Setup successful!')"
```

## Coding Standards

### Python Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

**Key guidelines:**
- Line length: 88 characters (Black-compatible)
- Use type hints for function signatures
- Write NumPy-style docstrings
- Follow PEP 8 conventions

### Pre-commit Checks

Pre-commit hooks automatically check:
- ✓ Ruff linting and formatting
- ✓ Spell checking (codespell)
- ✓ Documentation formatting

To run checks manually:

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Run only ruff
pre-commit run ruff --all-files

# Format code
ruff format eegdash/
```

### Docstrings

Use NumPy-style docstrings:

```python
def extract_features(dataset, extractors):
    """Extract features from an EEG dataset.

    Parameters
    ----------
    dataset : EEGDashDataset
        The dataset to extract features from.
    extractors : list of FeatureExtractor
        List of feature extractors to apply.

    Returns
    -------
    features : FeaturesDataset
        Dataset containing extracted features.

    Examples
    --------
    >>> from eegdash.features import SpectralFeatureExtractor
    >>> extractor = SpectralFeatureExtractor()
    >>> features = extract_features(dataset, [extractor])
    """
    ...
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=eegdash --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::test_eegdash_find -v
```

### Writing Tests

**Required:**
- All new features must include unit tests
- Bug fixes should include regression tests
- Aim for >80% code coverage

**Test structure:**

```python
import pytest
from eegdash import EEGDashDataset

def test_feature_description():
    """Test that feature does X when Y."""
    # Arrange
    dataset = create_test_dataset()

    # Act
    result = dataset.some_method()

    # Assert
    assert result is not None
    assert len(result) > 0
```

## Documentation

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -e .[docs]

# Build documentation
cd docs
make html

# View documentation
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

### Documentation Guidelines

- Update docstrings when changing function signatures
- Add usage examples to docstrings
- Update user guide (`docs/source/user_guide.rst`) for user-facing changes
- Update API documentation for new modules/classes

## Pull Request Process

### Submitting a Pull Request

1. **Push your branch:**
   ```bash
   git push origin feat/your-feature-name
   ```

2. **Open a pull request** on GitHub from your branch to `develop`

3. **Fill out the PR template** with:
   - Description of changes
   - Type of change (feature/bugfix/docs/etc.)
   - Testing performed
   - Related issues

4. **Respond to review feedback** promptly

5. **Update your PR** if needed:
   ```bash
   # Make changes
   git add .
   git commit -m "fix(feature): address review feedback"
   git push origin feat/your-feature-name
   ```

### PR Review Checklist

Reviewers will check:
- ✓ Code follows project style and conventions
- ✓ Tests are comprehensive and passing
- ✓ Documentation is clear and complete
- ✓ Commit messages follow conventions
- ✓ No breaking changes (or properly documented)
- ✓ Performance impact is acceptable

### After Approval

- PRs are merged by maintainers after approval
- Delete your branch after merging (done automatically)
- Update your local repository:
  ```bash
  git checkout develop
  git pull upstream develop
  ```

## Release Process

*For maintainers only*

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH` (e.g., `0.4.1`)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Creating a Release

1. **Update version** in `eegdash/__init__.py`
2. **Update CHANGELOG.md** with release notes
3. **Create a PR** from `develop` to `main`
4. **Tag the release** after merging:
   ```bash
   git tag -a v0.4.1 -m "Release version 0.4.1"
   git push origin v0.4.1
   ```
5. **PyPI publishing** happens automatically via GitHub Actions

## License

By contributing, you agree that your contributions will be licensed under the open source project license.

## Acknowledgments

EEG-DaSh is a collaborative initiative between:
- **Swartz Center for Computational Neuroscience (SCCN)**, UC San Diego
- **Ben-Gurion University (BGU)**, Israel

Supported by the **National Science Foundation (NSF)**.

---

**Thank you for contributing to EEG-Dash!** 🧠✨
