..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003987
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003987
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003987``
- **Summary:** Modality: Visual | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 23
- **Number of Recordings:** 69
- **Number of Tasks:** 1
- **Number of Channels:** 64
- **Sampling Frequencies:** 500.0930232558139
- **Total Duration (hours):** 52.076
- **Dataset Size:** 26.41 GB
- **OpenNeuro:** `ds003987 <https://openneuro.org/datasets/ds003987>`__
- **NeMAR:** `ds003987 <https://nemar.org/dataexplorer/detail?dataset_id=ds003987>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds003987        23       64           1     500.093         52.076  26.41 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003987

   dataset = DS003987(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003987>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003987>`__

