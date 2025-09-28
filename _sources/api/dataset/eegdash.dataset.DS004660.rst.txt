..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004660
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004660
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004660``
- **Summary:** Modality: Multisensory | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 21
- **Number of Recordings:** 42
- **Number of Tasks:** 1
- **Number of Channels:** 32
- **Sampling Frequencies:** 2048,512
- **Total Duration (hours):** 23.962
- **Dataset Size:** 7.25 GB
- **OpenNeuro:** `ds004660 <https://openneuro.org/datasets/ds004660>`__
- **NeMAR:** `ds004660 <https://nemar.org/dataexplorer/detail?dataset_id=ds004660>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes  Freq(Hz)      Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds004660        21       32           1  2048,512           23.962  7.25 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004660

   dataset = DS004660(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004660>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004660>`__

