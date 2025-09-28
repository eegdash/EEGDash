..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003516
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003516
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003516``
- **Summary:** Modality: Auditory | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 25
- **Number of Recordings:** 25
- **Number of Tasks:** 1
- **Number of Channels:** 47
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 22.57
- **Dataset Size:** 13.46 GB
- **OpenNeuro:** `ds003516 <https://openneuro.org/datasets/ds003516>`__
- **NeMAR:** `ds003516 <https://nemar.org/dataexplorer/detail?dataset_id=ds003516>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds003516        25       47           1         500          22.57  13.46 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003516

   dataset = DS003516(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003516>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003516>`__

