..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004952
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004952
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004952``
- **Summary:** Modality: Visual | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 10
- **Number of Recordings:** 245
- **Number of Tasks:** 1
- **Number of Channels:** 128
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 123.411
- **Dataset Size:** 696.72 GB
- **OpenNeuro:** `ds004952 <https://openneuro.org/datasets/ds004952>`__
- **NeMAR:** `ds004952 <https://nemar.org/dataexplorer/detail?dataset_id=ds004952>`__

=========  =======  =======  ==========  ==========  =============  =========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =========
ds004952        10      128           1        1000        123.411  696.72 GB
=========  =======  =======  ==========  ==========  =============  =========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004952

   dataset = DS004952(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004952>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004952>`__

