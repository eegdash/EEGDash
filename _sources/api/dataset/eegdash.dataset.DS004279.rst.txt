..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004279
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004279
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004279``
- **Summary:** Modality: Auditory | Type: Perception | Subjects: Healthy
- **Number of Subjects:** 56
- **Number of Recordings:** 60
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 53.729
- **Dataset Size:** 25.22 GB
- **OpenNeuro:** `ds004279 <https://openneuro.org/datasets/ds004279>`__
- **NeMAR:** `ds004279 <https://nemar.org/dataexplorer/detail?dataset_id=ds004279>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004279        56       64           1        1000         53.729  25.22 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004279

   dataset = DS004279(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004279>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004279>`__

