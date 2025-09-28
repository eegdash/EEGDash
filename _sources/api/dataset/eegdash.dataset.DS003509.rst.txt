..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003509
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003509
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003509``
- **Summary:** Modality: Visual | Type: Learning | Subjects: Parkinson's
- **Number of Subjects:** 56
- **Number of Recordings:** 84
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 48.535
- **Dataset Size:** 22.34 GB
- **OpenNeuro:** `ds003509 <https://openneuro.org/datasets/ds003509>`__
- **NeMAR:** `ds003509 <https://nemar.org/dataexplorer/detail?dataset_id=ds003509>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds003509        56       64           1         500         48.535  22.34 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003509

   dataset = DS003509(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003509>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003509>`__

