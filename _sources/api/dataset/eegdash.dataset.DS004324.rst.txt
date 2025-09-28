..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004324
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004324
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004324``
- **Summary:** Modality: Multisensory | Type: Affect | Subjects: Healthy
- **Number of Subjects:** 26
- **Number of Recordings:** 26
- **Number of Tasks:** 1
- **Number of Channels:** 28
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 19.216
- **Dataset Size:** 2.46 GB
- **OpenNeuro:** `ds004324 <https://openneuro.org/datasets/ds004324>`__
- **NeMAR:** `ds004324 <https://nemar.org/dataexplorer/detail?dataset_id=ds004324>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds004324        26       28           1         500         19.216  2.46 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004324

   dataset = DS004324(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004324>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004324>`__

