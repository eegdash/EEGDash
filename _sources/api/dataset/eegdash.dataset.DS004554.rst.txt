..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004554
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004554
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004554``
- **Summary:** Modality: Visual | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 16
- **Number of Recordings:** 16
- **Number of Tasks:** 1
- **Number of Channels:** 99
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 0.024
- **Dataset Size:** 8.79 GB
- **OpenNeuro:** `ds004554 <https://openneuro.org/datasets/ds004554>`__
- **NeMAR:** `ds004554 <https://nemar.org/dataexplorer/detail?dataset_id=ds004554>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds004554        16       99           1        1000          0.024  8.79 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004554

   dataset = DS004554(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004554>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004554>`__

