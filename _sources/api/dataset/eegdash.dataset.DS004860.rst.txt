..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004860
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004860
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004860``
- **Summary:** Modality: Auditory | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 31
- **Number of Recordings:** 31
- **Number of Tasks:** 1
- **Number of Channels:** 32
- **Sampling Frequencies:** 2048,512
- **Total Duration (hours):** 0.0
- **Dataset Size:** 3.79 GB
- **OpenNeuro:** `ds004860 <https://openneuro.org/datasets/ds004860>`__
- **NeMAR:** `ds004860 <https://nemar.org/dataexplorer/detail?dataset_id=ds004860>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes  Freq(Hz)      Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds004860        31       32           1  2048,512                0  3.79 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004860

   dataset = DS004860(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004860>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004860>`__

