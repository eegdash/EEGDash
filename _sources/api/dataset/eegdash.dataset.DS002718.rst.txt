..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS002718
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS002718
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS002718``
- **Summary:** Modality: Visual | Type: Perception | Subjects: Healthy
- **Number of Subjects:** 18
- **Number of Recordings:** 18
- **Number of Tasks:** 1
- **Number of Channels:** 74
- **Sampling Frequencies:** 250
- **Total Duration (hours):** 14.844
- **Dataset Size:** 4.31 GB
- **OpenNeuro:** `ds002718 <https://openneuro.org/datasets/ds002718>`__
- **NeMAR:** `ds002718 <https://nemar.org/dataexplorer/detail?dataset_id=ds002718>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds002718        18       74           1         250         14.844  4.31 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS002718

   dataset = DS002718(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds002718>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds002718>`__

