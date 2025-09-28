..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005342
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005342
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005342``
- **Summary:** Modality: Visual | Type: Motor | Subjects: Healthy
- **Number of Subjects:** 32
- **Number of Recordings:** 32
- **Number of Tasks:** 1
- **Number of Channels:** 17
- **Sampling Frequencies:** 250
- **Total Duration (hours):** 33.017
- **Dataset Size:** 2.03 GB
- **OpenNeuro:** `ds005342 <https://openneuro.org/datasets/ds005342>`__
- **NeMAR:** `ds005342 <https://nemar.org/dataexplorer/detail?dataset_id=ds005342>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds005342        32       17           1         250         33.017  2.03 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005342

   dataset = DS005342(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005342>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005342>`__

