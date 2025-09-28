..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003766
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003766
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003766``
- **Summary:** Modality: Visual | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 31
- **Number of Recordings:** 124
- **Number of Tasks:** 4
- **Number of Channels:** 129
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 39.973
- **Dataset Size:** 152.77 GB
- **OpenNeuro:** `ds003766 <https://openneuro.org/datasets/ds003766>`__
- **NeMAR:** `ds003766 <https://nemar.org/dataexplorer/detail?dataset_id=ds003766>`__

=========  =======  =======  ==========  ==========  =============  =========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =========
ds003766        31      129           4        1000         39.973  152.77 GB
=========  =======  =======  ==========  ==========  =============  =========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003766

   dataset = DS003766(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003766>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003766>`__

