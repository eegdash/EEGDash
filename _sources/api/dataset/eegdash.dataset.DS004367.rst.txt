..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004367
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004367
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004367``
- **Summary:** Modality: Visual | Type: Perception | Subjects: Schizophrenia/Psychosis
- **Number of Subjects:** 40
- **Number of Recordings:** 40
- **Number of Tasks:** 1
- **Number of Channels:** 68
- **Sampling Frequencies:** 1200
- **Total Duration (hours):** 24.81
- **Dataset Size:** 27.98 GB
- **OpenNeuro:** `ds004367 <https://openneuro.org/datasets/ds004367>`__
- **NeMAR:** `ds004367 <https://nemar.org/dataexplorer/detail?dataset_id=ds004367>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004367        40       68           1        1200          24.81  27.98 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004367

   dataset = DS004367(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004367>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004367>`__

