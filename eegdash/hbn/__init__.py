# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""Healthy Brain Network (HBN) specific utilities and preprocessing.

This module provides specialized functions for working with the Healthy Brain Network
dataset, including preprocessing pipelines, annotation handling, and windowing utilities
tailored for HBN EEG data analysis.
"""

from .preprocessing import hbn_ec_ec_reannotation
from .windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
    build_trial_table,
    keep_only_recordings_with,
)

__all__ = [
    "hbn_ec_ec_reannotation",
    "build_trial_table",
    "annotate_trials_with_target",
    "add_aux_anchors",
    "add_extras_columns",
    "keep_only_recordings_with",
]
