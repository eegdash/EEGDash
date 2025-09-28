..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005170
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005170
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005170``
- **Summary:** Modality: Visual | Type: other
- **Number of Subjects:** 5
- **Number of Recordings:** 225
- **Number of Tasks:** 1
- **Total Duration (hours):** 0.0
- **Dataset Size:** 261.77 GB
- **OpenNeuro:** `ds005170 <https://openneuro.org/datasets/ds005170>`__
- **NeMAR:** `ds005170 <https://nemar.org/dataexplorer/detail?dataset_id=ds005170>`__

=========  =======  =======  ==========  ==========  =============  =========
dataset      #Subj  #Chan      #Classes  Freq(Hz)      Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =========
ds005170         5                    1                          0  261.77 GB
=========  =======  =======  ==========  ==========  =============  =========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005170

   dataset = DS005170(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005170>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005170>`__

