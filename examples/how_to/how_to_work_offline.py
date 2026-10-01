"""Open a staged recording in an independent offline process
=========================================================

Stage one real 5.6 MB BNCI2014-004 recording online, then copy and run the
entire second code cell on the offline machine, including its imports.
EEGDash must already be installed. Keep the signal and BIDS sidecars together
under EEGDASH_CACHE_DIR. Running this whole script still requires a connection
for stage 1; only stage 2 is the standalone offline recipe.
Source: `nm000135 <https://nemar.org/dataset/nm000135>`_.
"""

# %%
# 1. Run once on a network-enabled staging machine
# ------------------------------------------------
from eegdash import EEGDashDataset
from eegdash.paths import get_default_cache_dir

online = EEGDashDataset(
    cache_dir=get_default_cache_dir(),
    dataset="nm000135",
    subject="1",
    session="0train",
    run="0",
    task="imagery",
    n_jobs=1,
)
online.download_all(n_jobs=1)
online.description[["subject", "session", "run", "task"]]

# %%
# 2. Standalone offline recipe: run this entire cell in a fresh process
# ---------------------------------------------------------------------
# Set EEGDASH_CACHE_DIR to the staged root before starting Python. This cell
# has no in-memory dependency on stage 1 and no network fallback. If discovery
# or opening fails, repair the complete staged BIDS tree on a connected host;
# do not create stand-in files or switch downloads on inside the compute job.
import pandas as pd

from eegdash import EEGDashDataset
from eegdash.paths import get_default_cache_dir

cache_dir = get_default_cache_dir()
query = dict(dataset="nm000135", subject="1", session="0train", run="0", task="imagery")
offline = EEGDashDataset(cache_dir=cache_dir, **query, download=False, n_jobs=1)
if len(offline.datasets) != 1:
    raise RuntimeError(
        "Expected one cached recording. Check cache root and BIDS sidecars."
    )
identity = offline.description.iloc[0]
for entity in ("subject", "session", "run", "task"):
    if str(identity[entity]) != query[entity]:
        raise ValueError(
            f"Cached {entity} differs from the requested {query[entity]!r}."
        )
raw = offline.datasets[0].raw
if not {"left_hand", "right_hand"} <= set(raw.annotations.description):
    raise ValueError(
        "Expected hand labels are missing; check the staged events sidecar."
    )
pd.DataFrame(
    [
        {
            **{
                entity: identity[entity]
                for entity in ("subject", "session", "run", "task")
            },
            "status": "opened locally",
            "channels": len(raw.ch_names),
            "sampling Hz": raw.info["sfreq"],
            "duration seconds": raw.n_times / raw.info["sfreq"],
        }
    ]
)

# %%
# Test network independence by disconnecting and running only stage 2.
# This demonstrates local loading, not offline installation or checkpoint
# acquisition: stage those dependencies separately. Cross-host checksums or
# a saved reference excerpt are optional transfer checks, not requirements
# for this independently runnable recipe. See the download how-to for the
# limited first-second reader-equivalence check.
