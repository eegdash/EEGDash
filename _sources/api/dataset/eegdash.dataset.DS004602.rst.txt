..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004602
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004602
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004602``
- **Summary:** Modality: Visual | Type: Perception | Subjects: Healthy
- **Number of Subjects:** 182
- **Number of Recordings:** 546
- **Number of Tasks:** 3
- **Number of Channels:** 128
- **Sampling Frequencies:** 250,500
- **Total Duration (hours):** 87.11
- **Dataset Size:** 73.91 GB
- **OpenNeuro:** `ds004602 <https://openneuro.org/datasets/ds004602>`__
- **NeMAR:** `ds004602 <https://nemar.org/dataexplorer/detail?dataset_id=ds004602>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes  Freq(Hz)      Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004602       182      128           3  250,500             87.11  73.91 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004602

   dataset = DS004602(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004602>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004602>`__

