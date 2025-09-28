..
   This file is auto-generated during the Sphinx build.
   Do not edit by hand; changes will be overwritten.

eegdash.dataset.DS004000
========================

.. currentmodule:: eegdash.dataset

.. autoclass:: eegdash.dataset.DS004000
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :member-order: bysource

Dataset Information
-------------------

- **Dataset ID:** ``DS004000``
- **Summary:** Modality: Multisensory | Type: Decision-making | Subjects: Schizophrenia/Psychosis
- **Number of Subjects:** 43
- **Number of Recordings:** 86
- **Number of Tasks:** 2
- **Number of Channels:** 128
- **Sampling Frequencies:** 2048
- **Total Duration (hours):** 0.0
- **Dataset Size:** 22.50 GB
- **OpenNeuro:** `ds004000 <https://openneuro.org/datasets/ds004000>`__
- **NeMAR:** `ds004000 <https://nemar.org/dataexplorer/detail?dataset_id=ds004000>`__

=========  =======  =======  ==========  ==========  =============  ========
dataset      #Subj    #Chan    #Classes    Freq(Hz)    Duration(H)  Size
=========  =======  =======  ==========  ==========  =============  ========
ds004000        43      128           2        2048              0  22.50 GB
=========  =======  =======  ==========  ==========  =============  ========


Usage Example
-------------

.. code-block:: python

   from eegdash.dataset import DS004000

   dataset = DS004000(cache_dir="./data")

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
* `OpenNeuro dataset page <https://openneuro.org/datasets/ds004000>`__
* `NeMAR dataset page <https://nemar.org/dataexplorer/detail?dataset_id=ds004000>`__

