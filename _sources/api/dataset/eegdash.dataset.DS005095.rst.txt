..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005095
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005095
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005095``
- **Summary:** Modality: Visual | Type: Memory | Subjects: Healthy
- **Number of Subjects:** 48
- **Number of Recordings:** 48
- **Number of Tasks:** 1
- **Number of Channels:** 63
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 16.901
- **Dataset Size:** 14.28 GB
- **OpenNeuro:** `ds005095 <https://openneuro.org/datasets/ds005095>`__
- **NeMAR:** `ds005095 <https://nemar.org/dataexplorer/detail?dataset_id=ds005095>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds005095        48       63           1        1000         16.901  14.28 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005095

   dataset = DS005095(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005095>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005095>`__

