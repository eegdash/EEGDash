..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS002578
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS002578
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS002578``
- **Summary:** Modality: Visual | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 2
- **Number of Recordings:** 2
- **Number of Tasks:** 1
- **Number of Channels:** 256
- **Sampling Frequencies:** 256
- **Total Duration (hours):** 1.455
- **Dataset Size:** 1.33 GB
- **OpenNeuro:** `ds002578 <https://openneuro.org/datasets/ds002578>`__
- **NeMAR:** `ds002578 <https://nemar.org/dataexplorer/detail?dataset_id=ds002578>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds002578         2      256           1         256          1.455  1.33 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS002578

   dataset = DS002578(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds002578>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds002578>`__

