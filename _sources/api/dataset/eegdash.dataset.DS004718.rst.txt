..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004718
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004718
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004718``
- **Summary:** Modality: Auditory | Type: Learning | Subjects: Healthy
- **Number of Subjects:** 51
- **Number of Recordings:** 51
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 21.836
- **Dataset Size:** 108.98 GB
- **OpenNeuro:** `ds004718 <https://openneuro.org/datasets/ds004718>`__
- **NeMAR:** `ds004718 <https://nemar.org/dataexplorer/detail?dataset_id=ds004718>`__

=========  =======  =======  ==========  ==========  =============  =========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =========
ds004718        51       64           1        1000         21.836  108.98 GB
=========  =======  =======  ==========  ==========  =============  =========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004718

   dataset = DS004718(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004718>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004718>`__

