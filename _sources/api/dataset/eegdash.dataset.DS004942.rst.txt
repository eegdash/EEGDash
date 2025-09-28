..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004942
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004942
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004942``
- **Summary:** Modality: Visual | Type: Memory | Subjects: Healthy
- **Number of Subjects:** 62
- **Number of Recordings:** 62
- **Number of Tasks:** 1
- **Number of Channels:** 65
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 28.282
- **Dataset Size:** 25.05 GB
- **OpenNeuro:** `ds004942 <https://openneuro.org/datasets/ds004942>`__
- **NeMAR:** `ds004942 <https://nemar.org/dataexplorer/detail?dataset_id=ds004942>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004942        62       65           1        1000         28.282  25.05 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004942

   dataset = DS004942(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004942>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004942>`__

