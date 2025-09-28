..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS003801
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS003801
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS003801``
- **Summary:** Modality: Auditory | Type: Attention | Subjects: Healthy
- **Number of Subjects:** 20
- **Number of Recordings:** 20
- **Number of Tasks:** 1
- **Number of Channels:** 24
- **Sampling Frequencies:** 250
- **Total Duration (hours):** 13.689
- **Dataset Size:** 1.15 GB
- **OpenNeuro:** `ds003801 <https://openneuro.org/datasets/ds003801>`__
- **NeMAR:** `ds003801 <https://nemar.org/dataexplorer/detail?dataset_id=ds003801>`__

=========  =======  =======  ==========  ==========  =============  =======
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  =======
ds003801        20       24           1         250         13.689  1.15 GB
=========  =======  =======  ==========  ==========  =============  =======


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS003801

   dataset = DS003801(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds003801>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds003801>`__

