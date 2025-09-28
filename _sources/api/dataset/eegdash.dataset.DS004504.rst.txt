..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004504
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004504
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004504``
- **Summary:** Modality: Resting State | Type: Clinical/Intervention | Subjects: Dementia
- **Number of Subjects:** 88
- **Number of Recordings:** 88
- **Number of Tasks:** 1
- **Number of Channels:** 19
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 19.608
- **Dataset Size:** 5.38 GB
- **OpenNeuro:** `ds004504 <https://openneuro.org/datasets/ds004504>`__
- **NeMAR:** `ds004504 <https://nemar.org/dataexplorer/detail?dataset_id=ds004504>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds004504        88       19           1         500         19.608  5.38 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004504

   dataset = DS004504(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004504>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004504>`__

