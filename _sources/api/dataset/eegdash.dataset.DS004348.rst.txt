..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004348
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004348
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004348``
- **Summary:** Modality: Sleep | Type: Sleep | Subjects: Healthy
- **Number of Subjects:** 9
- **Number of Recordings:** 18
- **Number of Tasks:** 2
- **Number of Channels:** 34
- **Sampling Frequencies:** 200
- **Total Duration (hours):** 35.056
- **Dataset Size:** 12.30 GB
- **OpenNeuro:** `ds004348 <https://openneuro.org/datasets/ds004348>`__
- **NeMAR:** `ds004348 <https://nemar.org/dataexplorer/detail?dataset_id=ds004348>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004348         9       34           2         200         35.056  12.30 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004348

   dataset = DS004348(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004348>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004348>`__

