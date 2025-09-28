..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004561
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004561
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004561``
- **Summary:** Modality: Motor | Type: Perception | Subjects: Healthy
- **Number of Subjects:** 23
- **Number of Recordings:** 23
- **Number of Tasks:** 1
- **Number of Channels:** 62
- **Sampling Frequencies:** 10000
- **Total Duration (hours):** 11.379
- **Dataset Size:** 97.96 GB
- **OpenNeuro:** `ds004561 <https://openneuro.org/datasets/ds004561>`__
- **NeMAR:** `ds004561 <https://nemar.org/dataexplorer/detail?dataset_id=ds004561>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004561        23       62           1       10000         11.379  97.96 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004561

   dataset = DS004561(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004561>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004561>`__

