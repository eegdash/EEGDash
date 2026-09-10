"""Keep gallery examples on recorded data without downloading in unit tests."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_gallery_has_no_simulated_data_or_score_generators():
    """Scan executable calls in every example, including private figure helpers."""
    # These constructors previously hid invented EEG, targets and benchmark
    # scores. Shuffling and selecting recorded observations remain allowed.
    generators = {
        "rand",
        "randn",
        "randint",
        "integers",
        "random_sample",
        "standard_normal",
        "normal",
        "uniform",
        "poisson",
        "binomial",
        "multivariate_normal",
        "make_classification",
        "make_regression",
        "make_blobs",
        "simulate_raw",
        "simulate_sparse_stc",
        "simulate_evoked",
        "make_cohort",
    }
    violations = []
    for path in sorted((ROOT / "examples").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            if name in generators or any(
                token in name.lower()
                for token in ("synthetic", "simulated", "fake_score")
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "Gallery data must come from recordings:\n" + "\n".join(
        violations
    )


def test_challenge_tutorials_load_eegdash_recordings():
    """Every published 2026 track must retain its real EEGDash loader."""
    scripts = sorted((ROOT / "examples" / "eeg2026").glob("tutorial_track_*.py"))
    assert len(scripts) == 4
    for path in scripts:
        tree = ast.parse(path.read_text(), filename=str(path))
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "EEGDashDataset"
            for node in ast.walk(tree)
        ), f"{path.name} must load recorded signals via EEGDashDataset"
