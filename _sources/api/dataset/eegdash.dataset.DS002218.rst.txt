..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS002218
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS002218
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS002218``
- **Summary:** Modality: Multisensory | Type: Perception | Subjects: Healthy
- **Number of Subjects:** 18
- **Number of Recordings:** 18
- **Number of Tasks:** 1
- **Number of Channels:** 0
- **Sampling Frequencies:** 256
- **Total Duration (hours):** 16.52
- **Dataset Size:** 1.95 GB
- **OpenNeuro:** `ds002218 <https://openneuro.org/datasets/ds002218>`__
- **NeMAR:** `ds002218 <https://nemar.org/dataexplorer/detail?dataset_id=ds002218>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds002218        18        0           1         256          16.52  1.95 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS002218

   dataset = DS002218(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds002218>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds002218>`__

