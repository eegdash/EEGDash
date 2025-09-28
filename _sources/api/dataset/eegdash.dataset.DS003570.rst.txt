..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003570
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003570
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003570``
- **Summary:** Modality: Auditory | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 40
- **Number of Recordings:** 40
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 2048
- **Total Duration (hours):** 26.208
- **Dataset Size:** 36.12 GB
- **OpenNeuro:** `ds003570 <https://openneuro.org/datasets/ds003570>`__
- **NeMAR:** `ds003570 <https://nemar.org/dataexplorer/detail?dataset_id=ds003570>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds003570        40       64           1        2048         26.208  36.12 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003570

   dataset = DS003570(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003570>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003570>`__

