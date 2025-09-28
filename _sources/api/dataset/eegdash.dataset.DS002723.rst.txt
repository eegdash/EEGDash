..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS002723
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS002723
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS002723``
- **Summary:** Modality: Auditory | Type: Affect | Subjects: Healthy
- **Number of Subjects:** 8
- **Number of Recordings:** 44
- **Number of Tasks:** 6
- **Number of Channels:** 32
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 0.0
- **Dataset Size:** 2.60 GB
- **OpenNeuro:** `ds002723 <https://openneuro.org/datasets/ds002723>`__
- **NeMAR:** `ds002723 <https://nemar.org/dataexplorer/detail?dataset_id=ds002723>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds002723         8       32           6        1000              0  2.60 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS002723

   dataset = DS002723(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds002723>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds002723>`__

