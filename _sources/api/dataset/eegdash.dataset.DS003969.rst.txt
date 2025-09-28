..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003969
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003969
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003969``
- **Summary:** Modality: Auditory | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 98
- **Number of Recordings:** 392
- **Number of Tasks:** 4
- **Number of Channels:** 64
- **Sampling Frequencies:** 1024,2048
- **Total Duration (hours):** 66.512
- **Dataset Size:** 54.46 GB
- **OpenNeuro:** `ds003969 <https://openneuro.org/datasets/ds003969>`__
- **NeMAR:** `ds003969 <https://nemar.org/dataexplorer/detail?dataset_id=ds003969>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes  Freq(Hz)      Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds003969        98       64           4  1024,2048          66.512  54.46 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003969

   dataset = DS003969(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003969>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003969>`__

