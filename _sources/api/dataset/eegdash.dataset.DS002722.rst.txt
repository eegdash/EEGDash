..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS002722
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS002722
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS002722``
- **Summary:** Modality: Auditory | Type: Affect | Subjects: Healthy
- **Number of Subjects:** 19
- **Number of Recordings:** 94
- **Number of Tasks:** 5
- **Number of Channels:** 32
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 0.0
- **Dataset Size:** 6.10 GB
- **OpenNeuro:** `ds002722 <https://openneuro.org/datasets/ds002722>`__
- **NeMAR:** `ds002722 <https://nemar.org/dataexplorer/detail?dataset_id=ds002722>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds002722        19       32           5        1000              0  6.10 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS002722

   dataset = DS002722(cache_dir="./data")

   print(f"Number of recordings: {len(dataset)}")

   if len(dataset):
       recording = dataset[0]
       raw = recording.load()
       print(f"Sampling rate: {raw.info['sfreq']} Hz")
       print(f"Channels: {len(raw.ch_names)}")


See Also
--------

* :class:`eegdash.dataset.EEGDashDataset`
* :mod:`eegdash.dataset`
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds002722>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds002722>`__

