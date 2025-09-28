..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003816
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003816
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003816``
- **Summary:** Modality: Other | Type: Affect | Subjects: Healthy
- **Number of Subjects:** 48
- **Number of Recordings:** 1077
- **Number of Tasks:** 8
- **Number of Channels:** 127
- **Sampling Frequencies:** 1000
- **Total Duration (hours):** 159.313
- **Dataset Size:** 53.97 GB
- **OpenNeuro:** `ds003816 <https://openneuro.org/datasets/ds003816>`__
- **NeMAR:** `ds003816 <https://nemar.org/dataexplorer/detail?dataset_id=ds003816>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds003816        48      127           8        1000        159.313  53.97 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003816

   dataset = DS003816(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003816>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003816>`__

