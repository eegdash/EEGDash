..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004796
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004796
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004796``
- **Summary:** Modality: Visual/Resting State | Type: Memory/Resting state | Subjects: Other
- **Number of Subjects:** 79
- **Number of Recordings:** 235
- **Number of Tasks:** 3
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 0.0
- **Dataset Size:** 240.21 GB
- **OpenNeuro:** `ds004796 <https://openneuro.org/datasets/ds004796>`__
- **NeMAR:** `ds004796 <https://nemar.org/dataexplorer/detail?dataset_id=ds004796>`__

=========  =======  =======  ==========  ==========  =============  =========
dataset      #Subj  #Chan      #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =========
ds004796        79                    3        1000              0  240.21 GB
=========  =======  =======  ==========  ==========  =============  =========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004796

   dataset = DS004796(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004796>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004796>`__

