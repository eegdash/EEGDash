..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003602
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003602
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003602``
- **Summary:** Modality: Visual | Type: Decision-making | Subjects: Other
- **Number of Subjects:** 118
- **Number of Recordings:** 699
- **Number of Tasks:** 6
- **Number of Channels:** 35
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 159.35
- **Dataset Size:** 73.21 GB
- **OpenNeuro:** `ds003602 <https://openneuro.org/datasets/ds003602>`__
- **NeMAR:** `ds003602 <https://nemar.org/dataexplorer/detail?dataset_id=ds003602>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds003602       118       35           6        1000         159.35  73.21 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003602

   dataset = DS003602(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003602>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003602>`__

