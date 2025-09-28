..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004621
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004621
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004621``
- **Summary:** Modality: Visual | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 42
- **Number of Recordings:** 167
- **Number of Tasks:** 4
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 0.0
- **Dataset Size:** 77.39 GB
- **OpenNeuro:** `ds004621 <https://openneuro.org/datasets/ds004621>`__
- **NeMAR:** `ds004621 <https://nemar.org/dataexplorer/detail?dataset_id=ds004621>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj  #Chan      #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004621        42                    4        1000              0  77.39 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004621

   dataset = DS004621(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004621>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004621>`__

