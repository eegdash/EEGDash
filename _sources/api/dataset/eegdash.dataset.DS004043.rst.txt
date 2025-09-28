..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004043
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004043
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004043``
- **Summary:** Modality: Visual | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 20
- **Number of Recordings:** 20
- **Number of Tasks:** 1
- **Number of Channels:** 63
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 0.0
- **Dataset Size:** 30.44 GB
- **OpenNeuro:** `ds004043 <https://openneuro.org/datasets/ds004043>`__
- **NeMAR:** `ds004043 <https://nemar.org/dataexplorer/detail?dataset_id=ds004043>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004043        20       63           1        1000              0  30.44 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004043

   dataset = DS004043(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004043>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004043>`__

