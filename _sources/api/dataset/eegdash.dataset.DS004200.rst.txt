..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004200
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004200
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004200``
- **Summary:** Modality: Multisensory | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 20
- **Number of Recordings:** 20
- **Number of Tasks:** 1
- **Number of Channels:** 37
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 14.123
- **Dataset Size:** 7.21 GB
- **OpenNeuro:** `ds004200 <https://openneuro.org/datasets/ds004200>`__
- **NeMAR:** `ds004200 <https://nemar.org/dataexplorer/detail?dataset_id=ds004200>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds004200        20       37           1        1000         14.123  7.21 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004200

   dataset = DS004200(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004200>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004200>`__

