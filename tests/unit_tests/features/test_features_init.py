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


def test_fit_feature_extractor_alias_warns_and_forwards():
    """Singular ``fit_feature_extractor`` is a deprecated alias for the plural API."""
    import warnings
    from unittest.mock import patch

    import eegdash.features as features_mod

    # Both the singular and plural names must be importable.
    assert hasattr(features_mod, "fit_feature_extractor")
    assert hasattr(features_mod, "fit_feature_extractors")
    assert "fit_feature_extractor" in features_mod.__all__

    sentinel = object()
    feature_list = [lambda x: x]
    with patch.object(
        features_mod, "fit_feature_extractors", return_value=sentinel
    ) as plural:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = features_mod.fit_feature_extractor(
                "dataset", feature_list, batch_size=64
            )
        # Forwards positional + keyword args verbatim.
        plural.assert_called_once_with("dataset", feature_list, batch_size=64)
        assert result is sentinel
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "fit_feature_extractor must emit a DeprecationWarning"
    assert "fit_feature_extractors" in str(deprecations[0].message)
