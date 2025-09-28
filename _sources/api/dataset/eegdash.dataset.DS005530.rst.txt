..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005530
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005530
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005530``
- **Number of Subjects:** 17
- **Number of Recordings:** 21
- **Number of Tasks:** 1
- **Number of Channels:** 10
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 154.833
- **Dataset Size:** 6.47 GB
- **OpenNeuro:** `ds005530 <https://openneuro.org/datasets/ds005530>`__
- **NeMAR:** `ds005530 <https://nemar.org/dataexplorer/detail?dataset_id=ds005530>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds005530        17       10           1         500        154.833  6.47 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005530

   dataset = DS005530(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005530>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005530>`__

