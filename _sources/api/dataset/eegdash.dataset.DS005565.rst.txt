..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005565
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005565
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005565``
- **Number of Subjects:** 24
- **Number of Recordings:** 24
- **Number of Tasks:** 1
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 11.436
- **Dataset Size:** 2.62 GB
- **OpenNeuro:** `ds005565 <https://openneuro.org/datasets/ds005565>`__
- **NeMAR:** `ds005565 <https://nemar.org/dataexplorer/detail?dataset_id=ds005565>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj  #Chan      #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds005565        24                    1         500         11.436  2.62 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005565

   dataset = DS005565(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005565>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005565>`__

