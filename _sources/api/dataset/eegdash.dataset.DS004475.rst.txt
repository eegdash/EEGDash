..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004475
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004475
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004475``
- **Summary:** Modality: Motor | Type: Motor | Subjects: Healthy
- **Number of Subjects:** 30
- **Number of Recordings:** 30
- **Number of Tasks:** 1
- **Number of Channels:** 113,115,118,119,120,122,123,124,125,126,127,128
- **Sampling Frequencies:** 512
- **Total Duration (hours):** 26.899
- **Dataset Size:** 112.74 GB
- **OpenNeuro:** `ds004475 <https://openneuro.org/datasets/ds004475>`__
- **NeMAR:** `ds004475 <https://nemar.org/dataexplorer/detail?dataset_id=ds004475>`__

=========  =======  ===============================================  ==========  ==========  =============  =========
dataset      #Subj  #Chan                                              #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  ===============================================  ==========  ==========  =============  =========
ds004475        30  113,115,118,119,120,122,123,124,125,126,127,128           1         512         26.899  112.74 GB
=========  =======  ===============================================  ==========  ==========  =============  =========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004475

   dataset = DS004475(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004475>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004475>`__

