..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003944
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003944
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003944``
- **Summary:** Modality: Resting State | Type: Clinical/Intervention | Subjects: Schizophrenia/Psychosis
- **Number of Subjects:** 82
- **Number of Recordings:** 82
- **Number of Tasks:** 1
- **Number of Channels:** 61
- **Sampling Frequencies:** 1000,3000.00030000003
- **Total Duration (hours):** 6.999
- **Dataset Size:** 6.15 GB
- **OpenNeuro:** `ds003944 <https://openneuro.org/datasets/ds003944>`__
- **NeMAR:** `ds003944 <https://nemar.org/dataexplorer/detail?dataset_id=ds003944>`__

=========  =======  =======  ==========  =====================  =============  =======
dataset      #Subj    #Chan    #Classes  Freq(Hz)                 Duration(H)  Size
=========  =======  =======  ==========  =====================  =============  =======
ds003944        82       61           1  1000,3000.00030000003          6.999  6.15 GB
=========  =======  =======  ==========  =====================  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003944

   dataset = DS003944(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003944>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003944>`__

