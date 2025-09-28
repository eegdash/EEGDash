..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005397
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005397
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005397``
- **Number of Subjects:** 26
- **Number of Recordings:** 26
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 27.923
- **Dataset Size:** 12.10 GB
- **OpenNeuro:** `ds005397 <https://openneuro.org/datasets/ds005397>`__
- **NeMAR:** `ds005397 <https://nemar.org/dataexplorer/detail?dataset_id=ds005397>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds005397        26       64           1         500         27.923  12.10 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005397

   dataset = DS005397(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005397>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005397>`__

