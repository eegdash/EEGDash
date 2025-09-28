..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004854
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004854
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004854``
- **Number of Subjects:** 1
- **Number of Recordings:** 1
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 128
- **Total Duration (hours):** 0.535
- **Dataset Size:** 79.21 MB
- **OpenNeuro:** `ds004854 <https://openneuro.org/datasets/ds004854>`__
- **NeMAR:** `ds004854 <https://nemar.org/dataexplorer/detail?dataset_id=ds004854>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004854         1       64           1         128          0.535  79.21 MB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004854

   dataset = DS004854(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004854>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004854>`__

