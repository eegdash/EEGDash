..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004444
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004444
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004444``
- **Summary:** Modality: Visual | Type: Motor | Subjects: Healthy
- **Number of Subjects:** 30
- **Number of Recordings:** 465
- **Number of Tasks:** 1
- **Number of Channels:** 129
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 55.687
- **Dataset Size:** 48.62 GB
- **OpenNeuro:** `ds004444 <https://openneuro.org/datasets/ds004444>`__
- **NeMAR:** `ds004444 <https://nemar.org/dataexplorer/detail?dataset_id=ds004444>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004444        30      129           1        1000         55.687  48.62 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004444

   dataset = DS004444(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004444>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004444>`__

