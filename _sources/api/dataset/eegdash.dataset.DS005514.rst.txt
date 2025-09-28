..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005514
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005514
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005514``
- **Number of Subjects:** 295
- **Number of Recordings:** 2885
- **Number of Tasks:** 10
- **Number of Channels:** 129
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 213.008
- **Dataset Size:** 185.03 GB
- **OpenNeuro:** `ds005514 <https://openneuro.org/datasets/ds005514>`__
- **NeMAR:** `ds005514 <https://nemar.org/dataexplorer/detail?dataset_id=ds005514>`__

=========  =======  =======  ==========  ==========  =============  =========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =========
ds005514       295      129          10         500        213.008  185.03 GB
=========  =======  =======  ==========  ==========  =============  =========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005514

   dataset = DS005514(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005514>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005514>`__

