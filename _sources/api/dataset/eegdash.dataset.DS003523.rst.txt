..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003523
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003523
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003523``
- **Summary:** Modality: Visual | Type: Memory | Subjects: TBI
- **Number of Subjects:** 91
- **Number of Recordings:** 221
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 84.586
- **Dataset Size:** 37.54 GB
- **OpenNeuro:** `ds003523 <https://openneuro.org/datasets/ds003523>`__
- **NeMAR:** `ds003523 <https://nemar.org/dataexplorer/detail?dataset_id=ds003523>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds003523        91       64           1         500         84.586  37.54 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003523

   dataset = DS003523(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003523>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003523>`__

