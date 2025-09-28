..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004117
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004117
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004117``
- **Summary:** Modality: Visual | Type: Memory | Subjects: Healthy
- **Number of Subjects:** 23
- **Number of Recordings:** 85
- **Number of Tasks:** 1
- **Number of Channels:** 69
- **Sampling Frequencies:** 1000,250,500,500.059
- **Total Duration (hours):** 15.941
- **Dataset Size:** 5.80 GB
- **OpenNeuro:** `ds004117 <https://openneuro.org/datasets/ds004117>`__
- **NeMAR:** `ds004117 <https://nemar.org/dataexplorer/detail?dataset_id=ds004117>`__

=========  =======  =======  ==========  ====================  =============  =======
dataset      #Subj    #Chan    #Classes  Freq(Hz)                Duration(H)  Size
=========  =======  =======  ==========  ====================  =============  =======
ds004117        23       69           1  1000,250,500,500.059         15.941  5.80 GB
=========  =======  =======  ==========  ====================  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004117

   dataset = DS004117(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004117>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004117>`__

