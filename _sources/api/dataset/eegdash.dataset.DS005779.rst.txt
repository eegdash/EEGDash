..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005779
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005779
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005779``
- **Number of Subjects:** 19
- **Number of Recordings:** 250
- **Number of Tasks:** 16
- **Number of Channels:** 64,67,70
- **Sampling Frequencies:** 5000
- **Total Duration (hours):** 16.65
- **Dataset Size:** 88.67 GB
- **OpenNeuro:** `ds005779 <https://openneuro.org/datasets/ds005779>`__
- **NeMAR:** `ds005779 <https://nemar.org/dataexplorer/detail?dataset_id=ds005779>`__

=========  =======  ========  ==========  ==========  =============  ========
dataset      #Subj  #Chan       #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  ========  ==========  ==========  =============  ========
ds005779        19  64,67,70          16        5000          16.65  88.67 GB
=========  =======  ========  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005779

   dataset = DS005779(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005779>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005779>`__

