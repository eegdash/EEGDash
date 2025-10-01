..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004951
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004951
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004951``
- **Summary:** Modality: Tactile | Type: Learning
- **Number of Subjects:** 11
- **Number of Recordings:** 23
- **Number of Tasks:** 1
- **Number of Channels:** 63
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 29.563
- **Dataset Size:** 22.00 GB
- **OpenNeuro:** `ds004951 <https://openneuro.org/datasets/ds004951>`__
- **NeMAR:** `ds004951 <https://nemar.org/dataexplorer/detail?dataset_id=ds004951>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004951        11       63           1        1000         29.563  22.00 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004951

   dataset = DS004951(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004951>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004951>`__

