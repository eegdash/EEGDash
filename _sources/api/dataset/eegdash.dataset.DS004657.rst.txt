..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004657
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004657
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004657``
- **Summary:** Modality: Motor | Type: Decision-making
- **Number of Subjects:** 24
- **Number of Recordings:** 119
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 1024,8192
- **Total Duration (hours):** 27.205
- **Dataset Size:** 43.06 GB
- **OpenNeuro:** `ds004657 <https://openneuro.org/datasets/ds004657>`__
- **NeMAR:** `ds004657 <https://nemar.org/dataexplorer/detail?dataset_id=ds004657>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes  Freq(Hz)      Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004657        24       64           1  1024,8192          27.205  43.06 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004657

   dataset = DS004657(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004657>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004657>`__

