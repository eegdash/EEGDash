..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005305
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005305
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005305``
- **Summary:** Modality: Visual | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 165
- **Number of Recordings:** 165
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 2048,512
- **Total Duration (hours):** 14.136
- **Dataset Size:** 6.41 GB
- **OpenNeuro:** `ds005305 <https://openneuro.org/datasets/ds005305>`__
- **NeMAR:** `ds005305 <https://nemar.org/dataexplorer/detail?dataset_id=ds005305>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes  Freq(Hz)      Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds005305       165       64           1  2048,512           14.136  6.41 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005305

   dataset = DS005305(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005305>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005305>`__

