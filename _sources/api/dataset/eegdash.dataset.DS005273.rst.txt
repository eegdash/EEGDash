..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005273
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005273
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005273``
- **Summary:** Modality: Visual | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 33
- **Number of Recordings:** 33
- **Number of Tasks:** 1
- **Number of Channels:** 63
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 58.055
- **Dataset Size:** 44.42 GB
- **OpenNeuro:** `ds005273 <https://openneuro.org/datasets/ds005273>`__
- **NeMAR:** `ds005273 <https://nemar.org/dataexplorer/detail?dataset_id=ds005273>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds005273        33       63           1        1000         58.055  44.42 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005273

   dataset = DS005273(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005273>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005273>`__

