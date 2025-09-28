..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004317
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004317
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004317``
- **Summary:** Modality: Multisensory | Type: Affect | Subjects: Healthy
- **Number of Subjects:** 50
- **Number of Recordings:** 50
- **Number of Tasks:** 1
- **Number of Channels:** 60
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 37.767
- **Dataset Size:** 18.29 GB
- **OpenNeuro:** `ds004317 <https://openneuro.org/datasets/ds004317>`__
- **NeMAR:** `ds004317 <https://nemar.org/dataexplorer/detail?dataset_id=ds004317>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004317        50       60           1         500         37.767  18.29 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004317

   dataset = DS004317(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004317>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004317>`__

