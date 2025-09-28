..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003751
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003751
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003751``
- **Summary:** Modality: Multisensory | Type: Affect | Subjects: Healthy
- **Number of Subjects:** 38
- **Number of Recordings:** 38
- **Number of Tasks:** 1
- **Number of Channels:** 128
- **Sampling Frequencies:** 250
- **Total Duration (hours):** 19.95
- **Dataset Size:** 4.71 GB
- **OpenNeuro:** `ds003751 <https://openneuro.org/datasets/ds003751>`__
- **NeMAR:** `ds003751 <https://nemar.org/dataexplorer/detail?dataset_id=ds003751>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds003751        38      128           1         250          19.95  4.71 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003751

   dataset = DS003751(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003751>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003751>`__

