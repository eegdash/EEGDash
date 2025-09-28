..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005131
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005131
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005131``
- **Summary:** Modality: Auditory | Type: Attention/Memory | Subjects: Healthy
- **Number of Subjects:** 58
- **Number of Recordings:** 63
- **Number of Tasks:** 2
- **Number of Channels:** 64
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 52.035
- **Dataset Size:** 22.35 GB
- **OpenNeuro:** `ds005131 <https://openneuro.org/datasets/ds005131>`__
- **NeMAR:** `ds005131 <https://nemar.org/dataexplorer/detail?dataset_id=ds005131>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds005131        58       64           2         500         52.035  22.35 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005131

   dataset = DS005131(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005131>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005131>`__

