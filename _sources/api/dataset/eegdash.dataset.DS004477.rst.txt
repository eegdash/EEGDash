..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004477
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004477
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004477``
- **Summary:** Modality: Multisensory | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 9
- **Number of Recordings:** 9
- **Number of Tasks:** 1
- **Number of Channels:** 79
- **Sampling Frequencies:** 2048
- **Total Duration (hours):** 13.557
- **Dataset Size:** 22.34 GB
- **OpenNeuro:** `ds004477 <https://openneuro.org/datasets/ds004477>`__
- **NeMAR:** `ds004477 <https://nemar.org/dataexplorer/detail?dataset_id=ds004477>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004477         9       79           1        2048         13.557  22.34 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004477

   dataset = DS004477(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004477>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004477>`__

