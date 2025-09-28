..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004460
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004460
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004460``
- **Summary:** Modality: Visual | Type: Perception | Subjects: Healthy
- **Number of Subjects:** 20
- **Number of Recordings:** 40
- **Number of Tasks:** 1
- **Number of Channels:** 160
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 27.494
- **Dataset Size:** 61.36 GB
- **OpenNeuro:** `ds004460 <https://openneuro.org/datasets/ds004460>`__
- **NeMAR:** `ds004460 <https://nemar.org/dataexplorer/detail?dataset_id=ds004460>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004460        20      160           1        1000         27.494  61.36 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004460

   dataset = DS004460(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004460>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004460>`__

