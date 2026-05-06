# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Human-readable summaries of cross-validation manifests.

:func:`describe_split` returns a small dictionary of per-fold statistics and,
optionally, prints a one-screen report. It is meant to be used in the
"split audit" cell of tutorials: confirm that folds are balanced enough,
catch tiny test sets, and surface session/site/dataset coverage problems
before training kicks off.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import pandas as pd

# Test folds smaller than this trigger a warning. The threshold is deliberately
# small because some neuroimaging splits (e.g. leave-one-subject-out across 4
# subjects) legitimately produce 1-subject test folds.
_MIN_TEST_GROUP_WARNING: int = 2


def _describe_fold(
    fold: dict[str, list[str]],
    metadata: pd.DataFrame,
    target: Optional[str],
) -> dict[str, Any]:
    """Compute per-fold statistics referenced by :func:`describe_split`."""
    train_ids = set(fold["train"])
    test_ids = set(fold["test"])

    if "sample_id" not in metadata.columns:
        raise ValueError(
            "metadata must have a 'sample_id' column to describe a manifest."
        )

    train_rows = metadata[metadata["sample_id"].isin(train_ids)]
    test_rows = metadata[metadata["sample_id"].isin(test_ids)]

    out: dict[str, Any] = {
        "n_train": len(train_rows.index),
        "n_test": len(test_rows.index),
        "subjects_train": int(train_rows.get("subject", pd.Series([])).nunique()),
        "subjects_test": int(test_rows.get("subject", pd.Series([])).nunique()),
        "sessions_train": int(train_rows.get("session", pd.Series([])).nunique()),
        "sessions_test": int(test_rows.get("session", pd.Series([])).nunique()),
        "datasets_train": int(train_rows.get("dataset", pd.Series([])).nunique()),
        "datasets_test": int(test_rows.get("dataset", pd.Series([])).nunique()),
    }

    if target is not None and target in metadata.columns:
        train_targets = train_rows[target].dropna().tolist()
        test_targets = test_rows[target].dropna().tolist()
        out["class_balance_train"] = dict(Counter(train_targets))
        out["class_balance_test"] = dict(Counter(test_targets))
    return out


def describe_split(
    manifest: dict[str, Any],
    metadata: pd.DataFrame,
    target: Optional[str] = None,
    print_report: bool = True,
) -> dict[str, Any]:
    """Return a structured summary of a manifest and optionally print a report.

    Parameters
    ----------
    manifest
        Manifest dictionary as returned by :func:`make_split_manifest`.
    metadata
        Per-sample metadata frame produced by :func:`to_split_metadata`.
    target
        Optional target column. When provided, the per-fold class balance is
        included in the returned summary.
    print_report
        When True (default) a short text report is printed to stdout. Pass
        False to keep stdout clean (e.g. inside test runs).

    Returns
    -------
    dict
        Keys: ``n_folds``, ``per_fold`` (list of dicts), ``coverage`` (overall
        subject/session/dataset counts), ``warnings`` (list of strings).

    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame.")
    folds = manifest.get("folds", [])
    per_fold = [_describe_fold(f, metadata, target) for f in folds]

    coverage = {
        "n_samples": int(len(metadata.index)),
        "n_subjects": int(metadata.get("subject", pd.Series([])).nunique()),
        "n_sessions": int(metadata.get("session", pd.Series([])).nunique()),
        "n_runs": int(metadata.get("run", pd.Series([])).nunique()),
        "n_datasets": int(metadata.get("dataset", pd.Series([])).nunique()),
    }

    warnings: list[str] = []
    for index, stats in enumerate(per_fold):
        if stats["n_test"] == 0:
            warnings.append(f"Fold {index} has an empty test set.")
        if stats["subjects_test"] < _MIN_TEST_GROUP_WARNING:
            warnings.append(
                f"Fold {index} test set covers only "
                f"{stats['subjects_test']} subject(s)."
            )
        if target is not None and "class_balance_test" in stats:
            classes = stats["class_balance_test"]
            if classes and len(classes) < 2:
                warnings.append(
                    f"Fold {index} test set has a single class "
                    f"({list(classes.keys())[0]!r}) -- chance level will be 100%."
                )

    summary = {
        "n_folds": int(manifest.get("n_folds", len(folds))),
        "splitter_class": manifest.get("splitter_class"),
        "random_seed": manifest.get("random_seed"),
        "target": manifest.get("target", target),
        "coverage": coverage,
        "per_fold": per_fold,
        "warnings": warnings,
    }

    if print_report:
        _print_report(summary)

    return summary


def _print_report(summary: dict[str, Any]) -> None:
    """Render a compact summary suitable for tutorials."""
    cov = summary["coverage"]
    print(
        "Split summary -- "
        f"folds={summary['n_folds']}, "
        f"splitter={summary['splitter_class']}, "
        f"seed={summary['random_seed']}, "
        f"target={summary['target']}"
    )
    print(
        f"Coverage: n_samples={cov['n_samples']}, "
        f"subjects={cov['n_subjects']}, sessions={cov['n_sessions']}, "
        f"runs={cov['n_runs']}, datasets={cov['n_datasets']}"
    )
    for index, stats in enumerate(summary["per_fold"]):
        line = (
            f"Fold {index}: "
            f"train={stats['n_train']} ({stats['subjects_train']} subj), "
            f"test={stats['n_test']} ({stats['subjects_test']} subj)"
        )
        if "class_balance_test" in stats:
            line += f", classes_test={stats['class_balance_test']}"
        print(line)
    if summary["warnings"]:
        print("Warnings:")
        for warning in summary["warnings"]:
            print(f"  - {warning}")


__all__ = ["describe_split"]
