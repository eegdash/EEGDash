..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004745
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004745
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004745``
- **Number of Subjects:** 6
- **Number of Recordings:** 6
- **Number of Tasks:** 1
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 0.0
- **Dataset Size:** 242.08 MB
- **OpenNeuro:** `ds004745 <https://openneuro.org/datasets/ds004745>`__
- **NeMAR:** `ds004745 <https://nemar.org/dataexplorer/detail?dataset_id=ds004745>`__

=========  =======  =======  ==========  ==========  =============  =========
dataset      #Subj  #Chan      #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =========
ds004745         6                    1        1000              0  242.08 MB
=========  =======  =======  ==========  ==========  =============  =========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004745

   dataset = DS004745(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004745>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004745>`__

