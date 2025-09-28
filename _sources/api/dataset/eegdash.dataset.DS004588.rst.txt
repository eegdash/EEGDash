..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004588
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004588
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004588``
- **Summary:** Modality: Visual | Type: Decision-making | Subjects: Healthy
- **Number of Subjects:** 42
- **Number of Recordings:** 42
- **Number of Tasks:** 1
- **Number of Channels:** 24
- **Sampling Frequencies:** 300
- **Total Duration (hours):** 4.957
- **Dataset Size:** 601.76 MB
- **OpenNeuro:** `ds004588 <https://openneuro.org/datasets/ds004588>`__
- **NeMAR:** `ds004588 <https://nemar.org/dataexplorer/detail?dataset_id=ds004588>`__

=========  =======  =======  ==========  ==========  =============  =========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =========
ds004588        42       24           1         300          4.957  601.76 MB
=========  =======  =======  ==========  ==========  =============  =========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004588

   dataset = DS004588(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004588>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004588>`__

