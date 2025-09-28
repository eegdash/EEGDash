..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005034
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005034
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005034``
- **Summary:** Modality: Visual | Type: Memory | Subjects: Healthy
- **Number of Subjects:** 25
- **Number of Recordings:** 100
- **Number of Tasks:** 2
- **Number of Channels:** 129
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 37.525
- **Dataset Size:** 61.36 GB
- **OpenNeuro:** `ds005034 <https://openneuro.org/datasets/ds005034>`__
- **NeMAR:** `ds005034 <https://nemar.org/dataexplorer/detail?dataset_id=ds005034>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds005034        25      129           2        1000         37.525  61.36 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005034

   dataset = DS005034(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005034>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005034>`__

