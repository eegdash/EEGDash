..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004015
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004015
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004015``
- **Summary:** Modality: Auditory | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 36
- **Number of Recordings:** 36
- **Number of Tasks:** 1
- **Number of Channels:** 18
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 47.29
- **Dataset Size:** 6.03 GB
- **OpenNeuro:** `ds004015 <https://openneuro.org/datasets/ds004015>`__
- **NeMAR:** `ds004015 <https://nemar.org/dataexplorer/detail?dataset_id=ds004015>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds004015        36       18           1         500          47.29  6.03 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004015

   dataset = DS004015(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004015>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004015>`__

