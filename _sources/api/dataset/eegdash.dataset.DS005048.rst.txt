..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005048
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005048
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005048``
- **Summary:** Modality: Auditory | Type: Attention | Subjects: Dementia
- **Number of Subjects:** 35
- **Number of Recordings:** 35
- **Number of Tasks:** 1
- **Sampling Frequencies:** 250
- **Total Duration (hours):** 5.203
- **Dataset Size:** 355.91 MB
- **OpenNeuro:** `ds005048 <https://openneuro.org/datasets/ds005048>`__
- **NeMAR:** `ds005048 <https://nemar.org/dataexplorer/detail?dataset_id=ds005048>`__

=========  =======  =======  ==========  ==========  =============  =========
dataset      #Subj  #Chan      #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =========
ds005048        35                    1         250          5.203  355.91 MB
=========  =======  =======  ==========  ==========  =============  =========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005048

   dataset = DS005048(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005048>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005048>`__

