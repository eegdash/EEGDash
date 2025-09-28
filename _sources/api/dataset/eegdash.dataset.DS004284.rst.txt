..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004284
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004284
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004284``
- **Summary:** Modality: Visual | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 18
- **Number of Recordings:** 18
- **Number of Tasks:** 1
- **Number of Channels:** 129
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 9.454
- **Dataset Size:** 16.49 GB
- **OpenNeuro:** `ds004284 <https://openneuro.org/datasets/ds004284>`__
- **NeMAR:** `ds004284 <https://nemar.org/dataexplorer/detail?dataset_id=ds004284>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004284        18      129           1        1000          9.454  16.49 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004284

   dataset = DS004284(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004284>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004284>`__

