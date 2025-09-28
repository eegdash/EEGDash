..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS001971
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS001971
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS001971``
- **Summary:** Modality: Auditory | Type: Motor | Subjects: Healthy
- **Number of Subjects:** 20
- **Number of Recordings:** 273
- **Number of Tasks:** 1
- **Number of Channels:** 108
- **Sampling Frequencies:** 512
- **Total Duration (hours):** 46.183
- **Dataset Size:** 31.98 GB
- **OpenNeuro:** `ds001971 <https://openneuro.org/datasets/ds001971>`__
- **NeMAR:** `ds001971 <https://nemar.org/dataexplorer/detail?dataset_id=ds001971>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds001971        20      108           1         512         46.183  31.98 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS001971

   dataset = DS001971(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds001971>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds001971>`__

