.. _concepts-metadata-and-bids:

Metadata and BIDS entities
==========================

EEG-BIDS is the contract that lets EEGDash advertise 700+ datasets as a single
queryable catalogue. Every recording the library knows about has been
described using the same vocabulary — *subject*, *session*, *task*, *run*,
acquisition system, electrode montage, units, sampling frequency — so that
filters work the same way regardless of the originating lab or repository.
This page explains what those entities mean, how they map to EEGDash query
keywords, and where participant-level metadata enters the picture.

The EEG-BIDS specification was published as Pernet et al. (2019) [1]_; if you
have not skimmed the paper, the short version is that BIDS standardizes file
names, directory layout, and accompanying JSON/TSV sidecars so that any tool
can walk a dataset without bespoke loaders. EEGDash reuses those entity names
verbatim in its query API.

BIDS entities at a glance
-------------------------

A BIDS-compliant EEG file path encodes the following entities (with the most
common ones in **bold**):

- **subject** (``sub-XYZ``): one participant.
- **session** (``ses-N``): one acquisition appointment for that participant.
  A subject can have multiple sessions when data are collected across days.
- **task** (``task-ABC``): the experimental paradigm (resting state,
  oddball, steady-state visual stimulation, etc.).
- **run** (``run-K``): a repetition of the same task within one session.
- *acquisition* (``acq-XYZ``): an acquisition variant, such as a different
  amplifier or montage.
- *processing* (``proc-XYZ``): a derivative pipeline tag.

The entities are hierarchical: a recording is uniquely identified by the
combination of dataset and the entities present in its filename. EEGDash's
catalogue stores these as separate fields in MongoDB, which is what makes
queries like "all resting-state runs from subject sub-002 of ds002718"
trivial to express.

Mapping BIDS entities to EEGDash query keywords
-----------------------------------------------

EEGDash's filter keywords are intentionally identical to the BIDS entity
names. The table below shows the mapping. A keyword accepts a single string
or a list of strings; lists are interpreted as MongoDB ``$in`` queries.

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - BIDS entity
     - EEGDash query keyword
     - Notes
   * - ``sub-<label>``
     - ``subject``
     - One label per recording. Pass a list to filter many subjects.
   * - ``ses-<label>``
     - ``session``
     - Optional. Many datasets have a single session and omit it.
   * - ``task-<label>``
     - ``task``
     - Required by BIDS for EEG. Use ``EEGDash.search_datasets(task=...)``
       for catalogue-level discovery.
   * - ``run-<index>``
     - ``run``
     - Optional. Several runs may share a task and session.
   * - dataset id
     - ``dataset``
     - The BIDS dataset identifier (e.g. ``ds002718``). Always required
       at the dataset object layer.
   * - modality
     - ``modality``
     - Catalogue-level: ``"eeg"``, ``"meg"``, ``"ieeg"``, ``"fnirs"``,
       ``"emg"``. Not a BIDS entity but stored in the same record.
   * - sampling rate
     - ``sampling_frequency``
     - From the BIDS sidecar JSON. Numeric.
   * - channel count
     - ``nchans``
     - From the channels TSV. Numeric.
   * - n samples
     - ``ntimes``
     - Sample count of the underlying file.

Anything not listed above is currently *not* a query field; if a record
attribute matters but is not selectable, please open an issue. The
authoritative list lives in the source as
``eegdash.const.ALLOWED_QUERY_FIELDS``.

Participant metadata and ``description_fields``
-----------------------------------------------

The entities above describe *recordings*. A different layer of metadata
describes the *participant*: age, sex, group, handedness, clinical
diagnosis, and so on. In BIDS this lives in ``participants.tsv`` next to
the dataset root. EEGDash exposes those columns through the
``description_fields`` argument of ``EEGDashDataset``.

For each recording, the dataset object builds a ``description`` row that
combines the BIDS entities (``subject``, ``session``, ``run``, ``task``)
with whatever participant fields you ask for:

.. code-block:: python

   from eegdash import EEGDashDataset

   ds = EEGDashDataset(
       cache_dir="./eegdash_cache",
       dataset="ds002718",
       description_fields=["subject", "session", "task", "age", "sex"],
   )
   print(ds.description.head())

This is the table you should plot, group by, and split on. Building a
braindecode-compatible ``WindowsDataset`` will copy these descriptors onto
each window automatically, so subject-aware splitters and metadata-driven
analyses both read from the same source of truth.

Why standardized metadata matters
---------------------------------

Two practical consequences fall out of using BIDS as the primary metadata
layer:

1. **Cross-dataset queries are cheap.** Asking "every resting-state
   recording with at least 64 channels and a sampling rate above 250 Hz"
   is a single MongoDB query because every dataset is indexed by the
   same fields. Without BIDS this would require writing 700+ ad hoc
   loaders.

2. **Splits are honest.** The leakage and evaluation pages
   (:doc:`leakage_and_evaluation`) rely on the fact that ``subject`` and
   ``session`` are reliably populated. A splitter that uses
   ``ds.description["subject"]`` as the grouping key only works because
   BIDS forces every dataset to expose that label.

If you load data from a non-BIDS source, you can still use ``EEGDashRaw``
manually — but you lose the ability to discover, split, and merge across
datasets. Spending an afternoon converting a folder of ``.set`` files to
BIDS is almost always worth it; see the EEG-BIDS paper [1]_ for the full
specification.

Metadata Glossary
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Definition
   * - **dataset**
     - The BIDS dataset identifier (e.g., ``ds004504``). Identifies the
       originating release.
   * - **subject**
     - A unique identifier for a participant within a dataset.
   * - **session**
     - A single appointment or day of recording for a subject.
   * - **task**
     - The experimental paradigm (e.g., ``RestingState``, ``visualoddball``).
   * - **run**
     - A repetition of the same task within one session.
   * - **age**
     - Participant age, typically in years at the time of the first session.
   * - **sex**
     - Biological status (typically ``M`` or ``F``) as recorded by the sponsor.
   * - **gender**
     - Participant's self-reported gender identity.
   * - **p_factor**
     - A transdiagnostic mental-health summary score derived from
       psychiatric questionnaires.
   * - **BIDS**
     - Brain Imaging Data Structure. The standard directory layout and
       naming convention for neural data.
   * - **windows**
     - Short, fixed-length segments of EEG data (e.g., 2 seconds) used as
       individual samples for machine learning.

Related tutorials
-----------------

- :doc:`/generated/auto_examples/tutorials/00_start_here/plot_00_first_search`
  uses BIDS entity filters end to end.
- :doc:`/generated/auto_examples/tutorials/00_start_here/plot_01_first_recording`
  shows how the recording-level metadata lines up with the actual data.
- :doc:`/generated/auto_examples/tutorials/10_core_workflow/plot_11_leakage_safe_split`
  consumes ``description["subject"]`` to build subject-aware splits.

Further reading
---------------

.. [1] Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G.,
   Phillips, C., Delorme, A., & Oostenveld, R. (2019). EEG-BIDS, an
   extension to the brain imaging data structure for electroencephalography.
   *Scientific Data*, 6(1), 103. https://doi.org/10.1038/s41597-019-0104-8

- Gorgolewski, K. J., et al. (2016). The brain imaging data structure, a
  format for organizing and describing outputs of neuroimaging
  experiments. *Scientific Data*, 3(1), 160044.
  https://doi.org/10.1038/sdata.2016.44
- Cisotto, G., & Chicco, D. (2024). Ten quick tips for clinical
  electroencephalographic (EEG) data acquisition and signal processing.
  *PeerJ Computer Science*, 10, e2256.
  https://doi.org/10.7717/peerj-cs.2256
