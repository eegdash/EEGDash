..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004603
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004603
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004603``
- **Summary:** Modality: Visual | Type: Perception | Subjects: Healthy
- **Number of Subjects:** 37
- **Number of Recordings:** 37
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 1024
- **Total Duration (hours):** 30.653
- **Dataset Size:** 39.13 GB
- **OpenNeuro:** `ds004603 <https://openneuro.org/datasets/ds004603>`__
- **NeMAR:** `ds004603 <https://nemar.org/dataexplorer/detail?dataset_id=ds004603>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004603        37       64           1        1024         30.653  39.13 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004603

   dataset = DS004603(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004603>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004603>`__

