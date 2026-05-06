# Contributing to EEG-Dash

Thanks for taking the time to contribute. This document covers the conventions, workflows, and tooling for working on EEG-Dash.

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
python -c "import eegdash; print(eegdash.__version__)"
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
- ✓ Tests cover the change and pass
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

## Tutorial Review Rubric (Reviewer-Only Rules)

The tutorial audit pipeline (`scripts/tutorial_audit/`) auto-validates most of the 49-rule rubric described in `docs/tutorial_implementation_strategy.md`. Eight rules cannot be machine-checked because they assess narrative quality, scientific framing, and hedging. A human reviewer (or, optionally, an LLM whose suggestion the human ratifies) scores them and commits the result alongside the tutorial's evidence dossier. The merge gate fails until every reviewer-only rule scores at least 3.

Cross-references:

- `docs/tutorial_implementation_strategy.md` — full rubric, validator implementations, and CI design.
- `docs/tutorial_restructure_plan.md` — the 13 tutorials, file layout, and quality bar.

### Reviewer-only rules

| Rule | Group | What you score |
| --- | --- | --- |
| E2.11 | Pedagogical | Single coherent narrative: opens with "why we care", proceeds through "how", closes with "what we found / next". No abandoned threads, no surprise digressions. |
| E2.14 | Pedagogical | Cognitive load is managed: each cell does one conceptual thing, long expressions are broken into named intermediates, every figure sits adjacent to the prose that explains it. |
| E2.17 | Pedagogical | At least one error is shown intentionally (a broken filter, a leaked split, a wrong shape) and recovered from in prose, modelling debugging behaviour rather than hiding it. |
| E4.31 | Engagement | The first five lines name a real neuroscience question, not a generic "this example shows class X". The reader should know what the tutorial is for before any import statement. |
| E4.33 | Engagement | The end result has scientific meaning: above-chance accuracy framed against chance level, an interpretable spectrum, an ERP that resembles literature. "Code runs" is not enough. |
| E4.35 | Engagement | Tone is "we"-inclusive, present tense, and explains the *why* of choices rather than the *what* of syntax. |
| E5.46 | Domain correctness | Claims are hedged, limitations are flagged. A small CI-friendly tutorial does not over-generalise from one subject or one task. |
| E6.47 | Diataxis | The document is unambiguously a tutorial — one rail, one outcome. Reference material is linked to API docs rather than embedded; how-to recipes are linked to a separate how-to. |

### Scoring grid (1-5)

Use the same grid for every reviewer-only rule. The grid is calibrated so a tutorial that "meets the bar" earns a 3 and a tutorial that an instructor would happily reuse earns a 4 or 5.

| Score | Label | Description |
| --- | --- | --- |
| 1 | Fails | Rule is clearly violated. Block merge until rewritten. |
| 2 | Below bar | Rule is partially addressed but a typical learner would notice the gap. Request changes. |
| 3 | Meets bar | Rule is satisfied. Acceptable for merge. |
| 4 | Strong | Rule is satisfied with deliberate craft beyond the minimum (worked example pulls in cited literature, the intentional error is genuinely instructive, etc.). |
| 5 | Exemplary | Could be lifted into the rubric documentation as a positive example. |

A score of 1 or 2 must come with a one-paragraph rationale. A score of 3 should still leave a one-sentence note.

### Reviewer score artifact

Commit the reviewer's scores into `docs/evidence/tutorials/<tutorial_id>/reviewer_score.json` as part of the review. The CI gate parses this file and checks every reviewer-only rule scores at least 3.

```json
{
  "tutorial_id": "plot_11_leakage_safe_split",
  "reviewer": "github-handle",
  "reviewed_at": "2026-05-06T18:34:11Z",
  "rubric_version": "1.0",
  "scores": {
    "E2.11": {"score": 4, "rationale": "Clear arc from leaky split through diagnosis to subject-aware split."},
    "E2.14": {"score": 3, "rationale": "Cells stay focused; one figure could move closer to its prose."},
    "E2.17": {"score": 4, "rationale": "Intentional leaky split is shown, diagnosed, then fixed."},
    "E4.31": {"score": 5, "rationale": "Opens with a real subject-overlap mistake from a published benchmark."},
    "E4.33": {"score": 4, "rationale": "Reports above-chance accuracy with chance level on the same axes."},
    "E4.35": {"score": 3, "rationale": "Tone is mostly inclusive; one section drifts into imperative voice."},
    "E5.46": {"score": 4, "rationale": "Limitations of single-task generalisation flagged in wrap-up."},
    "E6.47": {"score": 3, "rationale": "Stays a tutorial; one paragraph could be linked to the explanation page instead."}
  }
}
```

### Merge gate

Merge requires **every** reviewer-only rule scored at least 3, **and** zero error-level findings from the static and runtime stages. A score of 1 or 2 on any rule blocks merge; the reviewer files an issue or requests changes. Reviewers should also confirm the spec advances from `state: reviewed` to `state: merged` only via `make -f tutorials.mk tutorial-release`.

### Notes for LLM-assisted review

A reviewer may use an LLM to draft scores, but the human signs off. Record the model name and prompt template in the rationale so future audits can re-run the same evaluation.
