..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005106
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005106
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005106``
- **Summary:** Modality: Visual | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 42
- **Number of Recordings:** 42
- **Number of Tasks:** 1
- **Number of Channels:** 32
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 0.012
- **Dataset Size:** 12.62 GB
- **OpenNeuro:** `ds005106 <https://openneuro.org/datasets/ds005106>`__
- **NeMAR:** `ds005106 <https://nemar.org/dataexplorer/detail?dataset_id=ds005106>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds005106        42       32           1         500          0.012  12.62 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005106

   dataset = DS005106(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005106>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005106>`__

