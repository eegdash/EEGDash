..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS002814
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS002814
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS002814``
- **Summary:** Modality: Visual | Type: Perception | Subjects: Healthy
- **Number of Subjects:** 21
- **Number of Recordings:** 168
- **Number of Tasks:** 1
- **Number of Channels:** 68
- **Sampling Frequencies:** 1200
- **Total Duration (hours):** 0.0
- **Dataset Size:** 48.57 GB
- **OpenNeuro:** `ds002814 <https://openneuro.org/datasets/ds002814>`__
- **NeMAR:** `ds002814 <https://nemar.org/dataexplorer/detail?dataset_id=ds002814>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds002814        21       68           1        1200              0  48.57 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS002814

   dataset = DS002814(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds002814>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds002814>`__

