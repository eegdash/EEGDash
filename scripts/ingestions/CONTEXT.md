# Ingestion Context

This file names the domain concepts used by the ingestion package and
its CI workflows. Keep these terms stable when moving `scripts/ingestions`
to a separate repository.

## Terms

**Source**
A remote or local dataset origin that can produce a catalogue of EEGDash
candidate datasets. Current production Sources are OpenNeuro and NEMAR.
Zenodo, Figshare, OSF, SciDB, DataRN, EEGManyLabs, and local BIDS are
secondary Sources until their CI paths are production-active.

The NEMAR Source pulls from `data.nemar.org` (neuroschema v0.3+
catalogue + per-dataset `metadata.json`). The legacy `nemardatasets`
GitHub org scan was retired in favor of the hosted API; the
shared client lives in `_nemar_api.py`.

**Source Catalogue**
The JSON listing produced by Stage 1 for one Source, for example
`openneuro_datasets.json` or `nemar_datasets.json`. A Source Catalogue
is durable input to clone/digest. It is not the same as a cloned dataset
tree.

**Dataset Listings Repository**
The external repository that stores durable Source Catalogues, cloned
state where applicable, and the latest Digest Corpus for scheduled runs.
Today this is `eegdash/eegdash-dataset-listings`.

**Clone Workspace**
The working directory produced by Stage 2. It contains one directory per
Dataset, with either a local BIDS tree, a manifest, or both. The Clone
Workspace is operational scratch state, not the stable public interface
of the ingestion package.

**Digest Corpus**
The Stage 3 output directory consumed by validation and injection. For
each Dataset it contains:

- `<dataset_id>_dataset.json`
- `<dataset_id>_records.json`
- `<dataset_id>_montages.json`
- `<dataset_id>_summary.json`

The Digest Corpus is the most important file-based interface in the
pipeline. Stage 4 validates it. Stage 5 plans and performs injection
from it. Daily CI should make this interface explicit.

**Injection Plan**
The structured decision produced from a Digest Corpus before writing to
the Gateway: which Datasets, Records, and Montages would be written,
which Datasets are skipped by fingerprint, and which recoverable load
errors occurred.

**Gateway Writer**
The module that performs authenticated writes to the EEGDash API Gateway:
Dataset bulk writes, Record upserts, Montage upserts, optional stats
recompute, and related mutation endpoints. The Gateway Writer is the
external write seam for Stage 5.

**Daily CI Run**
A scheduled GitHub Actions run in the ingestion repository. Its default
job is a dry-run gate: fetch current Sources, refresh the Digest Corpus,
validate it, build an Injection Plan, and publish artifacts/reports. A
production write is a separate, explicitly approved path.

## Design Pressure

Moving ingestion to a separate repository changes the main design
pressure. The goal is no longer only "make the scripts importable"; the
goal is to make the scheduled operational workflow reliable enough to be
the primary production check.

That makes the Digest Corpus and Injection Plan more important than the
package layout itself. The package layout should support those two
interfaces, not replace them with a speculative generic pipeline
abstraction.
