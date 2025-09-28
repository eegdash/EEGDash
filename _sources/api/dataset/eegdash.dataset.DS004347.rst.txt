..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004347
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004347
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004347``
- **Summary:** Modality: Visual | Type: Perception | Subjects: Healthy
- **Number of Subjects:** 24
- **Number of Recordings:** 48
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 128,512
- **Total Duration (hours):** 6.389
- **Dataset Size:** 2.69 GB
- **OpenNeuro:** `ds004347 <https://openneuro.org/datasets/ds004347>`__
- **NeMAR:** `ds004347 <https://nemar.org/dataexplorer/detail?dataset_id=ds004347>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes  Freq(Hz)      Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds004347        24       64           1  128,512             6.389  2.69 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004347

   dataset = DS004347(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004347>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004347>`__

