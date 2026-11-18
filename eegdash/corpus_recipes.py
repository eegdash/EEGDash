# Authors: The EEGDash contributors.
# License: BSD-3-Clause
# Copyright the EEGDash contributors.

"""The recipes a corpus can be converted with, all defined here.

A :class:`Recipe` is every number the conversion applies to the signal. It has
no default values: a recipe is written out in full below, added to
:data:`RECIPES`, and from there written into each converted part, so a store
can be checked against the recipe it claims::

    python -m eegdash.corpus_recipes      # every recipe, as JSON

The rules that give a channel its position (the position bank, the cap
templates, the alignment of a dataset's ``electrodes.tsv``) live in
:mod:`eegdash.corpus_convert`; a change to them is a new ``version`` of the
recipes that use them.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any

# OpenEEGBench sources in the catalogue: BCIC-IV-2a, PhysioNet-MI, FACED, Mumtaz,
# Zyma, CHB-MIT, BCIC-2020-3. A model scored on them must not be pretrained on them.
HELD_OUT = frozenset(
    {
        "nm000139",
        "ds004362",
        "nm000112",
        "nm000114",
        "nm000109",
        "nm000110",
        "nm000113",
    }
)


@dataclass(frozen=True)
class Recipe:
    """The preprocessing a part was converted with; written into every part.

    Parameters
    ----------
    name, version : str, int
        What a store is checked against before parts are merged into it.
    sfreq : int
        Sampling rate of the samples, Hz. A slower recording is skipped.
    l_freq, h_freq : float
        Band-pass edges, Hz.
    h_freq_max_ratio : float
        The upper edge is lowered to this fraction of a recording's own rate
        when that is below ``h_freq``.
    resample_method : str
        The ``method`` of ``mne.io.Raw.resample``.
    window : int
        Timesteps per sample. A recording of *window* timesteps or fewer after
        resampling is skipped.
    min_channels : int
        A recording with fewer channels of known position is skipped.
    units : str
        Unit of the stored signal.
    refused : frozenset of str
        Dataset ids never converted: the benchmark's own sources.

    """

    name: str
    version: int
    sfreq: int
    l_freq: float
    h_freq: float
    h_freq_max_ratio: float
    resample_method: str
    window: int
    min_channels: int
    units: str
    refused: frozenset[str]

    def refuses(self, dataset_id: str) -> bool:
        """Whether *dataset_id*, or the same dataset under its other archive id, is refused."""
        ids = {dataset_id}
        if re.fullmatch(r"(ds|on)\d+", dataset_id):
            ids |= {"ds" + dataset_id[2:], "on" + dataset_id[2:]}
        return bool(ids & self.refused)

    def to_dict(self) -> dict[str, Any]:
        """The recipe as JSON-ready values."""
        return {**asdict(self), "refused": sorted(self.refused)}


# The recipe of the REVE paper (section 3.1.1), as eegdash-corpus-v0 was converted:
# 200 Hz, band-pass 0.5-99.5 Hz, samples of 10 s, microvolts, not normalised.
REVE = Recipe(
    name="reve",
    version=1,
    sfreq=200,
    l_freq=0.5,
    h_freq=99.5,
    h_freq_max_ratio=0.45,
    resample_method="polyphase",
    window=2000,
    min_channels=3,
    units="uV",
    refused=HELD_OUT,
)

RECIPES: dict[str, Recipe] = {r.name: r for r in (REVE,)}


def get_recipe(recipe: str | Recipe) -> Recipe:
    """A :class:`Recipe`, by name or as given."""
    if isinstance(recipe, Recipe):
        return recipe
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}: {sorted(RECIPES)}")
    return RECIPES[recipe]


def main() -> int:
    """Print every recipe as JSON."""
    print(json.dumps({n: r.to_dict() for n, r in RECIPES.items()}, indent=1))
    return 0


__all__ = ["HELD_OUT", "REVE", "RECIPES", "Recipe", "get_recipe", "main"]

if __name__ == "__main__":
    sys.exit(main())
