..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003517
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003517
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003517``
- **Summary:** Modality: Visual | Type: Learning | Subjects: Healthy
- **Number of Subjects:** 17
- **Number of Recordings:** 34
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 13.273
- **Dataset Size:** 6.48 GB
- **OpenNeuro:** `ds003517 <https://openneuro.org/datasets/ds003517>`__
- **NeMAR:** `ds003517 <https://nemar.org/dataexplorer/detail?dataset_id=ds003517>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds003517        17       64           1         500         13.273  6.48 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003517

   dataset = DS003517(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003517>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003517>`__

