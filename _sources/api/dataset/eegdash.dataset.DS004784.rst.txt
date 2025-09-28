..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004784
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004784
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004784``
- **Summary:** Modality: Motor | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 1
- **Number of Recordings:** 6
- **Number of Tasks:** 6
- **Number of Channels:** 128
- **Sampling Frequencies:** 512
- **Total Duration (hours):** 0.518
- **Dataset Size:** 10.82 GB
- **OpenNeuro:** `ds004784 <https://openneuro.org/datasets/ds004784>`__
- **NeMAR:** `ds004784 <https://nemar.org/dataexplorer/detail?dataset_id=ds004784>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004784         1      128           6         512          0.518  10.82 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004784

   dataset = DS004784(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004784>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004784>`__

