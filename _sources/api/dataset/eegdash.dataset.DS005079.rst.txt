..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS005079
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS005079
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS005079``
- **Summary:** Modality: Multisensory | Type: Affect | Subjects: Healthy
- **Number of Subjects:** 1
- **Number of Recordings:** 60
- **Number of Tasks:** 15
- **Number of Channels:** 65
- **Sampling Frequencies:** 500
- **Total Duration (hours):** 3.25
- **Dataset Size:** 1.68 GB
- **OpenNeuro:** `ds005079 <https://openneuro.org/datasets/ds005079>`__
- **NeMAR:** `ds005079 <https://nemar.org/dataexplorer/detail?dataset_id=ds005079>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds005079         1       65          15         500           3.25  1.68 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS005079

   dataset = DS005079(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds005079>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds005079>`__

