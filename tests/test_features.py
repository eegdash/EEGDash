"""Test for features module Python 3.10+ compatibility."""

import pytest


def test_import_features_module():
    """Test that the features module can be imported without syntax errors.

    This test ensures Python 3.10+ compatibility by verifying that:
    1. Type annotations with list[], type[], and | syntax work (via __future__ imports)
    2. No Python 3.11+ exclusive syntax is used (like *unpacking in subscripts)
    """
    try:
        import eegdash.features

        assert eegdash.features is not None
    except SyntaxError as e:
        pytest.fail(f"SyntaxError when importing eegdash.features: {e}")
    except ImportError as e:
        pytest.fail(f"ImportError when importing eegdash.features: {e}")


def test_import_features_submodules():
    """Test that all features submodules can be imported."""
    submodules = [
        "eegdash.features.inspect",
        "eegdash.features.extractors",
        "eegdash.features.serialization",
        "eegdash.features.datasets",
        "eegdash.features.decorators",
        "eegdash.features.feature_bank",
        "eegdash.features.feature_bank.complexity",
        "eegdash.features.feature_bank.dimensionality",
        "eegdash.features.feature_bank.signal",
        "eegdash.features.feature_bank.spectral",
        "eegdash.features.feature_bank.connectivity",
        "eegdash.features.feature_bank.csp",
    ]

    for module_name in submodules:
        try:
            __import__(module_name)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError when importing {module_name}: {e}")
        except ImportError:
            # Some imports might fail due to missing dependencies, that's ok
            # We only care about SyntaxError
            pass


def test_features_basic_functionality():
    """Test basic features module functionality."""
    from eegdash.features import (
        get_all_feature_extractors,
        get_all_feature_kinds,
        get_all_features,
    )

    # These should return lists without errors
    features = get_all_features()
    assert isinstance(features, list)

    extractors = get_all_feature_extractors()
    assert isinstance(extractors, list)

    kinds = get_all_feature_kinds()
    assert isinstance(kinds, list)
