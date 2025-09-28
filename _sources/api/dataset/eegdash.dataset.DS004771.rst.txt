..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004771
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004771
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004771``
- **Summary:** Modality: Visual | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 61
- **Number of Recordings:** 61
- **Number of Tasks:** 1
- **Number of Channels:** 34
- **Sampling Frequencies:** 256
- **Total Duration (hours):** 0.022
- **Dataset Size:** 1.36 GB
- **OpenNeuro:** `ds004771 <https://openneuro.org/datasets/ds004771>`__
- **NeMAR:** `ds004771 <https://nemar.org/dataexplorer/detail?dataset_id=ds004771>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds004771        61       34           1         256          0.022  1.36 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004771

   dataset = DS004771(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004771>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004771>`__

