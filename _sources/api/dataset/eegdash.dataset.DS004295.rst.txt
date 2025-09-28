..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004295
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004295
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004295``
- **Summary:** Modality: Multisensory | Type: Learning | Subjects: Healthy
- **Number of Subjects:** 26
- **Number of Recordings:** 26
- **Number of Tasks:** 1
- **Number of Channels:** 66
- **Sampling Frequencies:** 1024,512
- **Total Duration (hours):** 34.313
- **Dataset Size:** 31.51 GB
- **OpenNeuro:** `ds004295 <https://openneuro.org/datasets/ds004295>`__
- **NeMAR:** `ds004295 <https://nemar.org/dataexplorer/detail?dataset_id=ds004295>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes  Freq(Hz)      Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004295        26       66           1  1024,512           34.313  31.51 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004295

   dataset = DS004295(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004295>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004295>`__

