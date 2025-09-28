..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005697
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005697
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005697``
- **Number of Subjects:** 50
- **Number of Recordings:** 50
- **Number of Tasks:** 1
- **Number of Channels:** 65,69
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 77.689
- **Dataset Size:** 66.58 GB
- **OpenNeuro:** `ds005697 <https://openneuro.org/datasets/ds005697>`__
- **NeMAR:** `ds005697 <https://nemar.org/dataexplorer/detail?dataset_id=ds005697>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj  #Chan      #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds005697        50  65,69             1        1000         77.689  66.58 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005697

   dataset = DS005697(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005697>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005697>`__

