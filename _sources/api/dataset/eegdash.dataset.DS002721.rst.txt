..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS002721
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS002721
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS002721``
- **Summary:** Modality: Auditory | Type: Affect | Subjects: Healthy
- **Number of Subjects:** 31
- **Number of Recordings:** 185
- **Number of Tasks:** 6
- **Number of Channels:** 19
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 0.0
- **Dataset Size:** 3.35 GB
- **OpenNeuro:** `ds002721 <https://openneuro.org/datasets/ds002721>`__
- **NeMAR:** `ds002721 <https://nemar.org/dataexplorer/detail?dataset_id=ds002721>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds002721        31       19           6        1000              0  3.35 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS002721

   dataset = DS002721(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds002721>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds002721>`__

