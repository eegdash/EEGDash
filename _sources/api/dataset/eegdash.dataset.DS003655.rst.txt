..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003655
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003655
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003655``
- **Summary:** Modality: Visual | Type: Memory | Subjects: Healthy
- **Number of Subjects:** 156
- **Number of Recordings:** 156
- **Number of Tasks:** 1
- **Number of Channels:** 19
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 130.923
- **Dataset Size:** 20.26 GB
- **OpenNeuro:** `ds003655 <https://openneuro.org/datasets/ds003655>`__
- **NeMAR:** `ds003655 <https://nemar.org/dataexplorer/detail?dataset_id=ds003655>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds003655       156       19           1         500        130.923  20.26 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003655

   dataset = DS003655(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003655>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003655>`__

