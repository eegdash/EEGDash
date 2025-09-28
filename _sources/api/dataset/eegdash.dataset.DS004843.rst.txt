..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004843
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004843
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004843``
- **Summary:** Modality: Visual | Type: Attention
- **Number of Subjects:** 14
- **Number of Recordings:** 92
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 256
- **Total Duration (hours):** 29.834
- **Dataset Size:** 7.66 GB
- **OpenNeuro:** `ds004843 <https://openneuro.org/datasets/ds004843>`__
- **NeMAR:** `ds004843 <https://nemar.org/dataexplorer/detail?dataset_id=ds004843>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds004843        14       64           1         256         29.834  7.66 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004843

   dataset = DS004843(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004843>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004843>`__

