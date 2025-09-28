..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS002720
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS002720
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS002720``
- **Summary:** Modality: Auditory | Type: Affect | Subjects: Healthy
- **Number of Subjects:** 18
- **Number of Recordings:** 165
- **Number of Tasks:** 10
- **Number of Channels:** 19
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 0.0
- **Dataset Size:** 2.39 GB
- **OpenNeuro:** `ds002720 <https://openneuro.org/datasets/ds002720>`__
- **NeMAR:** `ds002720 <https://nemar.org/dataexplorer/detail?dataset_id=ds002720>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds002720        18       19          10        1000              0  2.39 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS002720

   dataset = DS002720(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds002720>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds002720>`__

